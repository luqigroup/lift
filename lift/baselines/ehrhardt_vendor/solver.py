"""Vendored primal-dual solvers from Ehrhardt/Mukherjee/Wong (2025).

Source: ``solver.py``, ``utils.py`` (``power`` and ``proj_Lrelu`` only) and the
salt-and-pepper primal-dual loop of ``example_denoise.py`` from
``hsw43/icnn_primal_dual``, commit ``bb0d0bd``, Zenodo DOI
10.5281/zenodo.17426033 (MIT license). Vendored bit-for-bit, so that
reconstruction quality is attributable to the training recipe and not to solver
drift.

Patched against upstream:

1. ``from skimage.metrics import peak_signal_noise_ratio`` replaced by the local
   :func:`~lift.baselines.ehrhardt_vendor.physics.compare_psnr`, the identical
   formula for their no-``data_range`` call. Only the ``progress=True`` branch
   uses it.
2. ``power`` extracted verbatim from their ``utils.py``, severing that module's
   top-level ``odl`` import from this path.
3. ``soft_thresh`` never shipped upstream. It is rewritten here from the prox of
   ``tau * ||x - y||_1``: ``prox(v) = y + sign(v - y) * max(|v - y| - tau, 0)``.
4. ``PDHG_ICNN_L1`` lifts the inline salt-and-pepper loop of
   ``example_denoise.py`` into a function, with two generalizations: the
   epigraph projection goes through :func:`proj_epi`, whose non-smoothed branch
   is their ``proj_Lrelu``, and the ``v2`` update takes the Huber form from
   ``PDHG_ICNN`` when ``regularizer.icnn.smoothed``. Both reduce bit-for-bit to
   their script for the non-smoothed prior it shipped with.

Upstream quirks preserved deliberately, since the shipped checkpoint was
validated through exactly this code:

* In ``PDHG_ICNN``, ``xold, zold = x, z`` and ``xbar, zbar = x, z`` alias the
  same tensors, so the over-relaxation step ``xbar = 2x - xold`` evaluates to
  ``x`` and the solver runs without extrapolation. The L1 loop below rebinds
  ``x`` each iteration and therefore does extrapolate.
* In ``PDHG_ICNN``, ``v2`` is ``expand``-ed over the batch, so for batch > 1 the
  per-sample dual writes alias one memory row. Run the solver at batch 1.
* ``proj_epi`` differs cosmetically between their ``solver.py`` and the inline
  copy in ``example_inpaint.py``; the ``solver.py`` version is vendored here.
"""

import torch
import torch.nn.functional as F

from lift.baselines.ehrhardt_vendor.physics import compare_psnr


def cubic(a, b, c, d):
    f = ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0
    g = (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a ** 2.0)) + (27.0 * d / a)) / 27.0
    h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

    R = -(g / 2.0) + torch.sqrt(h)
    S = torch.sign(R) * torch.pow(torch.abs(R), 1 / 3.0)
    T = -(g / 2.0) - torch.sqrt(h)
    U = torch.sign(T) * torch.pow(torch.abs(T), 1 / 3.0)

    return (S + U) - (b / (3.0 * a))


# Projection to epigraph
def proj_epi(x0, x1, reg):
    alpha = reg.icnn.negative_slope
    if reg.icnn.smoothed:
        omega = reg.icnn.smoothing
        lower = -x0 / alpha
        mid = -x0 + omega * (3 + alpha) / 2

        act2_x0 = reg.icnn.act2(x0)

        cond1 = (x1 < alpha * x0) & (x1 <= lower)
        cond2 = (act2_x0 > x1) & (x1 > lower) & (x1 < mid)
        cond3 = (act2_x0 > x1) & (x1 >= mid)

        o1_case1 = (x0 + alpha * x1) / (1 + alpha ** 2)
        o2_case1 = alpha * o1_case1

        a = (1 - alpha) ** 2 / (2 * omega ** 2)
        b = 3 * alpha * (1 - alpha) / (2 * omega)
        c = 1 + alpha ** 2 - (1 - alpha) * x1 / omega
        d = -(x0 + alpha * x1)
        o1_case2 = cubic(a, b, c, d)
        o2_case2 = reg.icnn.act2(o1_case2)

        o1_case3 = (x0 + x1) / 2. + (1 - alpha) * omega / 4
        o2_case3 = (x1 + x0) / 2. - (1 - alpha) * omega / 4
        o1 = torch.where(cond1, o1_case1,
                torch.where(cond2, o1_case2,
                torch.where(cond3, o1_case3, x0)))
        o2 = torch.where(cond1, o2_case1,
                torch.where(cond2, o2_case2,
                torch.where(cond3, o2_case3, x1)))
    else:
        o_zero = torch.zeros_like(x0)
        # Condition masks
        cond1 = torch.abs(x1) < x0
        cond2 = (x1 < alpha * x0) & (x1 < -x0 / alpha)
        cond3 = (x1 >= -x0 / alpha) & (x1 <= -x0)
        # Precompute values
        o_temp1 = (x0 + x1) / 2.
        o_temp2 = (x0 + alpha * x1) / (1 + alpha ** 2)
        # Use torch.where for vectorized assignment
        o1 = torch.where(cond1, o_temp1,
                    torch.where(cond2, o_temp2,
                    torch.where(cond3, o_zero, x0)))
        o2 = torch.where(cond1, o_temp1,
                    torch.where(cond2, alpha*o_temp2,
                    torch.where(cond3, o_zero, x1)))
    return o1, o2


#Proj to epi (leaky_relu) -- verbatim from their utils.py
def proj_Lrelu(x_in, negative_slope):
    x0, x1 = x_in[0], x_in[1]
    o_zero = torch.zeros_like(x0)
    # Condition masks
    cond1 = torch.abs(x1) < x0
    cond2 = (x1 < negative_slope * x0) & (x1 < -x0 / negative_slope)
    cond3 = (x1 >= -x0 / negative_slope) & (x1 <= -x0)
    # Precompute values
    o_temp1 = (x0 + x1) / 2.
    o_temp2 = (x0 + negative_slope * x1) / (1 + negative_slope ** 2)
    # Use torch.where for vectorized assignment
    o1 = torch.where(cond1, o_temp1,
                torch.where(cond2, o_temp2,
                torch.where(cond3, o_zero, x0)))
    o2 = torch.where(cond1, o_temp1,
                torch.where(cond2, negative_slope*o_temp2,
                torch.where(cond3, o_zero, x1)))
    return [o1, o2]


#Power method -- verbatim from their utils.py (odl import severed)
def power(K_op, K_op_T, x_in, max_iter=100):
    with torch.no_grad():
        xk = x_in
        ss = []
        for _ in range(max_iter):
            Kxk = K_op(xk)
            xk = K_op_T(Kxk)
            xk = xk / torch.norm(xk)
            Kxk = K_op(xk)
            s = torch.norm(Kxk).item()
            ss.append(s)
            if len(ss) > 1 and ((ss[-1] - ss[-2]) ** 2 / ss[-2] ** 2 < 1e-6):
                break
        return s


def soft_thresh(x, y, tau):
    """Prox of ``tau * ||. - y||_1`` at ``x``."""
    d = x - y
    return y + torch.sign(d) * torch.clamp(torch.abs(d) - tau, min=0.0)


def PDHG_ICNN(
        x0,
        y,
        x_gt,
        physics,
        data_fidelity,
        regularizer,
        lamda,
        sigma0,
        sigma1,
        taux,
        tauz,
        max_iter,
        tol,
        device,
        verbose=False,
        progress=False,
):

    """Primal-dual hybrid gradient for ``min_x 0.5 ||Ax - y||_2^2 + lamda reg(x)``.

    Parameters:
    - x0: initial guess.
    - y: observed data.
    - x_gt: ground truth, used only for the progress log.
    - physics: forward operator, with ``A`` and ``A_adjoint`` methods.
    - data_fidelity: data-fidelity term.
    - regularizer: regularizer, wrapping an ICNN.
    - lamda: regularization parameter.
    - sigma0, sigma1, taux, tauz: primal and dual step sizes.
    - max_iter, tol: iteration budget and convergence tolerance.
    - device: device to run on.
    - verbose, progress: print and record per-iteration diagnostics.

    Returns the reconstruction, the iteration count and the final residual, with
    the progress log prepended when ``progress`` is set.
    """

    x = x0.detach().clone()
    z = regularizer.icnn.act2(regularizer.icnn.wx(x)).detach()
    b0 = regularizer.icnn.wx.bias.data.view(1,regularizer.icnn.n_filters,1,1)
    bf1 = regularizer.icnn.fc1.bias.data
    a_weight = lamda*regularizer.icnn.fc2.weight.data
    omega = regularizer.icnn.smoothing
    cmin = 0*a_weight
    cmax = a_weight
    mu = 0
    regmu = lamda*mu

    if progress:
        logs = {
            'loss': [],
            'data_fid': [],
            'reg': [],
            'psnr': [],
        }

    idx = torch.arange(0,x.shape[0],device=device)
    res = (tol+1)*torch.ones(x.shape[0],device=device)
    mask = physics.mask.data


    v11, v12 = torch.zeros_like(z), torch.zeros_like(z)
    v2 = lamda * regularizer.icnn.fc2.weight.data.clone()
    v2 = v2.expand(x.shape[0], -1)  # (batch, ...)
    v2[regularizer.icnn.WA(z)+bf1<=0] = 0
    xold, zold = x, z
    xbar, zbar = x, z

    if progress:
        reg = lamda*regularizer.g(x).detach().squeeze(1)
        data_fid = data_fidelity(x,y,physics).detach()
        loss = data_fid+reg
        logs['loss'].append(loss.item())
        logs['data_fid'].append(data_fid.item())
        logs['reg'].append(reg.item())
        logs['psnr'].append(compare_psnr(x_gt.cpu().numpy(), x.cpu().numpy()))

    for iter in range(max_iter):
        with torch.no_grad():
            v_in1, v_in2 = v11[idx]+sigma0*regularizer.icnn.W0(xbar[idx]), v12[idx]+sigma0*zbar[idx]
            projv11, projv12= proj_epi(v_in1/sigma0+b0,v_in2/sigma0,regularizer)
            v11[idx], v12[idx] = v_in1-sigma0*(projv11-b0), v_in2-sigma0*projv12
            if regularizer.icnn.smoothed:
                v2[idx] = torch.clamp((a_weight*(v2[idx]+sigma1*(regularizer.icnn.WA(zbar[idx])+bf1)))/(sigma1*omega+a_weight),cmin,cmax) # huber
            else:
                v2[idx] = torch.clamp(v2[idx]+sigma1*regularizer.icnn.WA(zbar[idx])+sigma1*bf1,cmin,cmax)

            x[idx] = (x[idx]-taux*regularizer.icnn.W0T(v11[idx])+taux*physics.A_adjoint(y[idx]))/(1+taux*regmu+taux*mask)
            z[idx] = z[idx]-tauz*(v12[idx]+regularizer.icnn.WAT(v2[idx]))

            xbar[idx], zbar[idx] = 2*x[idx]-xold[idx], 2*z[idx]-zold[idx]
            xold[idx], zold[idx] = x[idx], z[idx]

        if iter > 0:
            grad = (data_fidelity.grad(x[idx], y[idx], physics) + lamda * regularizer.grad(x[idx])).detach()
            res[idx] = torch.norm(grad,p=2,dim=(1,2,3))

        condition = res >= tol
        idx = condition.nonzero().view(-1)

        if torch.max(res) < tol:
            if verbose:
                print('Convergence reached at iteration {:d}'.format(iter+1))
            break


        if progress:
            reg = lamda*(regularizer.g(x)).detach().squeeze(1)
            data_fid = data_fidelity(x,y,physics).detach()
            loss = data_fid+reg
            logs['loss'].append(loss.item())
            logs['data_fid'].append(data_fid.item())
            logs['reg'].append(reg.item())
            logs['psnr'].append(compare_psnr(x_gt.cpu().numpy(), x.cpu().numpy()))
            if verbose:
                recon_log = '[iter: {:d}/{:d},var_loss: {:.10f},fid: {:.10f},reg: {:.10f}]'\
                        .format(iter+1, max_iter, loss.item(), data_fid.item(),reg.item())
                print(recon_log)

    return (x, logs, iter+1, torch.max(res)) if progress else (x, iter+1, torch.max(res))


def PDHG_ICNN_L1(
        y,
        regularizer,
        reg_param,
        c1,
        c2,
        max_iter,
        device,
):
    """L1-fidelity primal-dual loop for salt-and-pepper denoising:

    min_x reg_param * ||x - y||_1 + reg(x)

    Lifted from the inline loop of their ``example_denoise.py``, which it
    reproduces bit-for-bit for a non-smoothed prior, and run on a single image
    ``y`` of shape (1, 1, H, W). Their best step-size cell is c1=0.1, c2=5e-5,
    reg_param=0.02 and 200 iterations.

    Returns the reconstruction ``x``, of the same shape as ``y``.
    """
    icnn = regularizer.icnn
    with torch.no_grad():
        b0 = icnn.wx.bias.data.view(1, icnn.n_filters, 1, 1)
        bf1 = icnn.fc1.bias.data
        a_weight = icnn.fc2.weight.data
        omega = icnn.smoothing
        cmin = 0 * a_weight
        cmax = a_weight

        x = y.detach().clone().to(device)
        z = icnn.act2(icnn.wx(x)).detach()

        W0norm = power(icnn.W0, icnn.W0T, x)
        WAnorm = power(icnn.WA, icnn.WAT, z)
        sigma0 = c1 / W0norm ** 2
        sigma1 = c2 / WAnorm ** 2
        taux = 1 / (sigma0 * W0norm ** 2)
        tauz = 1 / (sigma0 + sigma1 * WAnorm ** 2)

        v1 = [torch.zeros_like(z), torch.zeros_like(z)]
        v_init = icnn.fc2.weight.data.clone()
        v_init[icnn.WA(z) + bf1 <= 0] = 0
        v2 = [v_init]
        v = [v1, v2]
        xold = x.clone()
        zold = z.clone()
        ubar = [2 * x - xold, 2 * z - zold]

        for iter in range(max_iter):
            v_in1 = [a + sigma0 * b
                     for a, b in zip(v[0], [icnn.W0(ubar[0]), ubar[1]])]
            projv1 = proj_epi(
                v_in1[0] / sigma0 + b0, v_in1[1] / sigma0, regularizer,
            )
            v[0][0] = v_in1[0] - sigma0 * (projv1[0] - b0)
            v[0][1] = v_in1[1] - sigma0 * projv1[1]
            if icnn.smoothed:
                v[1][0] = torch.clamp(
                    (a_weight * (v[1][0] + sigma1 * (icnn.WA(ubar[1]) + bf1)))
                    / (sigma1 * omega + a_weight),
                    cmin, cmax,
                )  # huber
            else:
                v[1][0] = torch.clamp(
                    (v[1][0] + sigma1 * icnn.WA(ubar[1])) + sigma1 * bf1,
                    cmin, cmax,
                )
            x = soft_thresh(
                x - taux * icnn.W0T(v[0][0]), y, taux * reg_param,
            )
            z = z - tauz * (v[0][1] + icnn.WAT(v[1][0]))
            ubar = [2 * x - xold, 2 * z - zold]
            xold = x
            zold = z

    return x
