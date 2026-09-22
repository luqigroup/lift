"""Input-convex neural network (ICNN) for parameterizing convex energies.

The ICNN outputs a scalar ``E(x)`` that is convex in ``x``: the
hidden-to-hidden weights are non-negative, the activations are convex and
non-decreasing (softplus or ReLU), and the input skip connections are
unconstrained. A strong-convexity quadratic is added so that
``q(x) = exp(-E(x)) / Z`` is well defined even before training.

``pos_constraint_mode`` selects how the hidden-to-hidden weights are kept
non-negative:

- ``"clamp"`` (default): the weights are stored raw and ``clamp_weights()``
  projects them onto the non-negative orthant after each optimizer step.
  This is the mode the EBM template inside the hypernetwork stack must use,
  since the softplus readouts already emit non-negative weights and
  ``clamp_weights`` is then a no-op.
- ``"softplus"``: the weights are reparameterized as ``W = softplus(W_raw)``,
  with no projection step. Used for direct-training baselines.

Three optional arguments turn the same module into a fully-connected convex
classifier: ``output_size`` for the class count, ``use_skip=False`` for the
no-skip architecture, and ``output_bias=True`` for the output-layer bias.
Every output coordinate is then convex in the input. The strong-convexity
term adds the same quadratic to every coordinate, so with ``output_size > 1``
it shifts all logits equally and leaves a softmax objective unchanged; the
classifier runs it at zero.

References:
    Amos, Xu and Kolter, "Input convex neural networks", ICML 2017.
    Hoedt and Klambauer, "Principled weight initialization for input-convex
        neural networks", NeurIPS 2023 (folded-normal init).
"""

import torch
import torch.nn.functional as F
from torch import nn


def _folded_normal_pos_init(shape: tuple[int, ...]) -> torch.Tensor:
    """Folded-normal init for non-negative weights (Hoedt-Klambauer 2023)."""
    n_in = shape[-1]
    sigma = (torch.pi / (2.0 * n_in)) ** 0.5
    return torch.abs(torch.randn(*shape) * sigma)


class ICNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        nlayers: int = 3,
        activation: str = "softplus",
        strong_convexity: float = 0.1,
        pos_constraint_mode: str = "clamp",
        output_size: int = 1,
        use_skip: bool = True,
        output_bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.strong_convexity = strong_convexity
        self.nlayers = nlayers
        self.output_size = int(output_size)
        self.use_skip = bool(use_skip)
        if pos_constraint_mode not in ("clamp", "softplus"):
            raise ValueError(
                f"unknown pos_constraint_mode: {pos_constraint_mode}"
            )
        self.pos_constraint_mode = pos_constraint_mode

        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.first_layer = nn.Linear(input_size, hidden_dim)

        # Hidden-to-hidden (non-negative). Stored raw; the effective weight
        # returned by ``_z_weight(i)`` applies pos_constraint_mode.
        self.z_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers - 1)]
        )
        # Input skip connections (unconstrained). Empty when ``use_skip`` is
        # False; the forward pass then omits the per-layer and output skips.
        self.x_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_dim, bias=False)
             for _ in range(nlayers - 1)] if self.use_skip else []
        )
        # Output head: positive combination + (optional) input skip.
        self.output_layer = nn.Linear(
            hidden_dim, self.output_size, bias=bool(output_bias),
        )
        self.output_x_layer = (
            nn.Linear(input_size, self.output_size, bias=False)
            if self.use_skip else None
        )

        # Tag the parameters that must stay non-negative so that
        # ``HyperNetwork.infer_pos_param_names`` can find them. The tag lives
        # on the ``nn.Parameter`` object's __dict__, not on the tensor data,
        # so it affects no computation and survives ``.to(device)`` and
        # state-dict round-trips. The tagged set matches
        # ``icnn_pos_param_names(nlayers)``.
        for layer in self.z_layers:
            layer.weight._pos_required = True
        self.output_layer.weight._pos_required = True

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        for layer in self.z_layers:
            if self.pos_constraint_mode == "softplus":
                init_pos = _folded_normal_pos_init(layer.weight.shape)
                # softplus^{-1}(y) = log(exp(y) - 1); use expm1 for stability.
                layer.weight.data = torch.log(
                    torch.expm1(init_pos.clamp(min=1e-6))
                )
            else:
                layer.weight.data = _folded_normal_pos_init(layer.weight.shape)
            layer.bias.data.zero_()

        if self.pos_constraint_mode == "softplus":
            init_pos = _folded_normal_pos_init(self.output_layer.weight.shape)
            self.output_layer.weight.data = torch.log(
                torch.expm1(init_pos.clamp(min=1e-6))
            )
        else:
            self.output_layer.weight.data = _folded_normal_pos_init(
                self.output_layer.weight.shape
            )

        if self.output_layer.bias is not None:
            self.output_layer.bias.data.zero_()

        nn.init.xavier_uniform_(self.first_layer.weight)
        self.first_layer.bias.data.zero_()
        for layer in self.x_layers:
            nn.init.xavier_uniform_(layer.weight)
        if self.output_x_layer is not None:
            nn.init.xavier_uniform_(self.output_x_layer.weight)

    # ------------------------------------------------------------------ effective weights
    def _z_weight(self, i: int) -> torch.Tensor:
        raw = self.z_layers[i].weight
        if self.pos_constraint_mode == "clamp":
            return raw  # caller is responsible for clamping after .step()
        return F.softplus(raw)

    def _out_weight(self) -> torch.Tensor:
        raw = self.output_layer.weight
        if self.pos_constraint_mode == "clamp":
            return raw
        return F.softplus(raw)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.first_layer(x))
        for i in range(len(self.z_layers)):
            W = self._z_weight(i)
            h = F.linear(z, W, self.z_layers[i].bias)
            if self.use_skip:
                h = h + self.x_layers[i](x)
            z = self.activation(h)
        Wout = self._out_weight()
        energy = F.linear(z, Wout, self.output_layer.bias)
        if self.output_x_layer is not None:
            energy = energy + self.output_x_layer(x)
        if self.strong_convexity > 0.0:
            energy = energy + 0.5 * self.strong_convexity * (x ** 2).sum(
                dim=-1, keepdim=True
            )
        return energy

    # ------------------------------------------------------------------ projection
    def clamp_weights(self) -> None:
        """Project the hidden-to-hidden weights to the non-negative orthant.

        No-op in ``softplus`` mode, where the parameterization handles
        positivity.
        """
        if self.pos_constraint_mode != "clamp":
            return
        for layer in self.z_layers:
            layer.weight.data.clamp_(min=0.0)
        self.output_layer.weight.data.clamp_(min=0.0)

    # ------------------------------------------------------------------ utilities
    def shift(self, b: torch.Tensor) -> "ShiftedICNN":
        return ShiftedICNN(self, b)


class ShiftedICNN(nn.Module):
    def __init__(self, icnn: ICNN, b: torch.Tensor) -> None:
        super().__init__()
        self.icnn = icnn
        self.register_buffer("b", b.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.icnn(x - self.b)

    def clamp_weights(self) -> None:
        self.icnn.clamp_weights()
