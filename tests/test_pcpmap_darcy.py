"""Contract tests for the PCP-Map Darcy experiment.

The load-bearing one is ``test_emitted_weights_do_not_couple_samples``.
``PCPMap.gxinv`` differentiates the potential through ``x`` to build the
transport map, so if the hypernetwork's emitted weights are a live
function of the same batch tensor, that derivative picks up the path
through the DeepSets mean and every sample's map value comes to depend
on every other sample in the batch -- while the log-determinant term,
computed under ``torch.func``, still sees the weights as constants. The
two halves of the likelihood would then be derivatives of different
potentials. The training loop guards against this by summarizing a
detached copy of the batch; this test pins that behaviour down, along
with the fact that detaching does not sever the gradient into the
emitter's own parameters.
"""

from __future__ import annotations

import torch

from lift.baselines.pcpmap import PCPMap
from lift.baselines.picnn import PICNN
from lift.baselines.picnn_hypernet import PicnnHypernetYConditional


def _build(d_x: int = 6, d_y: int = 3, batch: int = 8):
    torch.manual_seed(0)
    picnn = PICNN(
        input_x_dim=d_x,
        input_y_dim=d_y,
        feature_dim=8,
        feature_y_dim=4,
        out_dim=1,
        num_layers=2,
        pos_constraint_mode="softplus_reparam",
    )
    hypernet = PicnnHypernetYConditional(
        picnn.lw_weight_shapes,
        input_x_dim=d_x,
        input_y_dim=d_y,
        cond_dim=4,
        summary_dim=8,
        phi_hidden=8,
        phi_out=8,
        rho_hidden=8,
        hidden_sizes="16",
        pos_constraint_mode="softplus_reparam",
    )
    prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(d_x), torch.ones(d_x)), 1,
    )
    model = PCPMap(prior, picnn, pos_weights_supplier=hypernet.emit)
    x = torch.randn(batch, d_x)
    y = torch.randn(batch, d_y)
    return model, hypernet, x, y


def test_emitted_weights_do_not_couple_samples():
    """The transport map of one sample must not depend on the others."""
    model, hypernet, x, y = _build()
    hypernet.prepare(x.detach(), y)
    xg = x.detach().clone().requires_grad_(True)
    z = model.gxinv(xg, y)
    # Backpropagate from sample 0's map value only; no other row of the
    # batch may pick up a gradient.
    (grad,) = torch.autograd.grad(z[0].sum(), xg, retain_graph=True)
    assert grad[0].abs().sum() > 0.0, "sample 0 must depend on itself"
    assert torch.allclose(
        grad[1:], torch.zeros_like(grad[1:]),
    ), "emitted weights leaked a cross-sample dependence into the map"


def test_detached_summary_still_trains_the_emitter():
    """Detaching the summary must not sever the emitter's own gradients."""
    model, hypernet, x, y = _build()
    hypernet.prepare(x.detach(), y)
    xg = x.detach().clone().requires_grad_(True)
    model.loss(xg, y).backward()
    grads = [p.grad for p in hypernet.parameters()]
    assert all(g is not None for g in grads), "an emitter parameter got no grad"
    total = sum(float((g ** 2).sum()) for g in grads)
    assert total > 0.0, "emitter received an identically zero gradient"


def test_cone_mode_projection_restores_feasibility():
    """``project_positive`` must leave no negative convex-path weight."""
    torch.manual_seed(0)
    picnn = PICNN(
        input_x_dim=4, input_y_dim=2, feature_dim=8, feature_y_dim=4,
        out_dim=1, num_layers=2, pos_constraint_mode="cone",
    )
    with torch.no_grad():
        picnn.Lw[0].weight.data.fill_(-1.0)
    picnn.project_positive()
    assert all(
        float(lw.weight.data.min()) >= 0.0 for lw in picnn.Lw
    ), "cone projection left a negative weight"
