# lift

Code, data and machine-checked proofs for

> **A lift for input-convex neural net training.**
> Ali Siahkoohi. Preprint, 2026.

## Overview

Input-convex neural nets need non-negative inter-layer weights. A softplus positivity map
attenuates the gradient where the latent weight is negative, so a coordinate that reaches that
shoulder stays. The **lift** emits that weight from a summary of the training batch instead,
coupling it to the gradient formed on it.

This repository reproduces the paper's five experiments and its theory:

| experiment | what the ICNN is | driver |
|---|---|---|
| one-dimensional gallery | a log-concave energy-based model on four targets | `scripts/experiments_lcmm_1d_logconcave.py` |
| tabular density estimation | the same, on POWER, HEPMASS and MiniBooNE | `scripts/experiments_pgd_uci_hepmass.py` |
| convex potential flows | the potential of a normalizing flow on two-dimensional toys | `scripts/experiments_cpflow_toy_published.py` |
| groundwater transport map | a partially convex map for a Bayesian inverse problem | `scripts/experiments_pcpmap_darcy.py` |
| image regularizer | a learned convex regularizer for inpainting at 256x256 | `scripts/experiments_ffhq_icnn_regularizer.py` |

The theoretical results are machine-checked in Lean 4 (see [Formal verification](#formal-verification)).

## Installation

```bash
git clone https://github.com/luqigroup/lift
cd lift
pip install -e .
```

Python 3.10+ with PyTorch. A GPU is needed to train anything at the sizes the paper reports; the
tests and the worked example below run on a CPU in seconds.

## Using the lift

The lift applies to any module whose constrained weights are marked. The ICNNs in this package
mark their own, so applying it is one line and needs no download and no training.

```python
import torch
from lift import ICNN, Lift

icnn = ICNN(input_size=2, hidden_dim=64, nlayers=3)
model = Lift(icnn, cond_size=2)          # the body reads the batch, the readout emits the weights

x = torch.randn(256, 2)
energy = model(x)                        # weights emitted from the same batch that forms the loss

for name in sorted(model.pos_param_names):
    print(f"{name:20s} min {model.weights(x)[name].min().item():.4f}")

energy.mean().backward()                 # the gradient reaches the lift; the ICNN stays frozen
```

Every emitted constrained weight is positive, so the target is convex in its input at each step.
`model.parameters()` is exactly the body and the slack.

To lift a module of your own, mark the weights convexity depends on:

```python
from lift import Lift, tag_positive_

tag_positive_(my_icnn, ["layer1.weight", "layer2.weight"])
model = Lift(my_icnn, cond_size=d)
```

`Lift` refuses a module with no marked weights rather than silently emitting through no positivity
map at all. `scripts/lift_example.py` is this example as a runnable file.

## Data

No trained weights ship here. Every result is reproduced by running the experiment, and every
input is public and downloaded on first use by the script that needs it.

| experiment | input | source |
|---|---|---|
| one-dimensional gallery | none, the targets are sampled analytically | — |
| tabular density estimation | POWER, HEPMASS, MiniBooNE | Zenodo record 1161203, the benchmark of Papamakarios et al. (2017) |
| convex potential flows | none, the toys are sampled analytically | — |
| groundwater transport map | fifty thousand training pairs and two thousand evaluation pairs | fetched like the others, or regenerated with `scripts/prepare_darcy_kl.py`, which needs [priorlaundermat](https://github.com/alisiahkoohi/priorlaundermat) |
| image regularizer | grayscale FFHQ pairs and the inpainting mask | Zenodo DOI 10.5281/zenodo.17426033, the release of Ehrhardt et al. (2026) |

The FFHQ imagery inside that archive is NVIDIA's, under CC BY-NC-SA 4.0. It is fetched from its
publisher and never redistributed here.

To pre-fetch a tier instead of letting it stream in:

```python
from lift.download import ensure_tier, sources
print(sources())
ensure_tier("uci")
ensure_tier("ffhq")
ensure_tier("darcy")
```

Destinations come from `projorg`, which locates the project itself, so nothing in this repository
hard-codes a path. Preparing the tabular splits from the downloaded archive:

```bash
python scripts/prepare_uci_power.py
python scripts/prepare_uci_hepmass.py
python scripts/prepare_uci_miniboone.py
```

Fetching and unpacking the image data:

```bash
python scripts/prepare_ffhq.py
```

## Recipes

Each experiment is reported at one configuration per construction, and they are not all the
default. The one-dimensional driver defaults to the reported recipe. On the tabular benchmarks
pass the flags below, where a ramp raises the step on the energy and the sampler flag raises it on
the sampler as well.

| | POWER | HEPMASS | MiniBooNE |
|---|---|---|---|
| lift | none | `--n_ebm_warmup_iters N --hyper_pool_norm 1` | `--n_ebm_warmup_iters N --sampler_lr_ramp 1 --hyper_pool_norm 1` |
| PGD | `--n_ebm_warmup_iters N` | none | `--n_ebm_warmup_iters N --sampler_lr_ramp 1` |
| direct softplus | none | none | none |

The one-dimensional recipe checks out on a single seed: Gumbel at the reported configuration gives
a total-variation distance of 0.011 against the 0.015 of the run the paper reports, and 0.149 with
the pooled summary left unstandardized.

## Reproducing the paper's figures

Each figure is the end of a chain: prepare the data, run the experiment across its seeds, then
render. The renderers read the run directories `projorg` writes, so a figure follows its
experiment without any path being passed by hand.

| figure | renderer |
|---|---|
| one-dimensional densities and training objective | `scripts/render_1d_panel.py` |
| tabular training objective | `scripts/render_tabular_val_loss_multiseed.py` |
| dwell on the shoulder | `scripts/render_dwell_survival.py` |
| convex potential flow densities | `scripts/render_cpflow_toy_panel.py` |
| groundwater posterior and objective | `scripts/render_darcy_panel.py` |
| image regularizer resilience | `scripts/render_ffhq_resilience.py` |
| image regularizer reconstructions | `scripts/render_ffhq_panel.py` |
| held-out read along training | `scripts/render_train_loss_appendix.py --curve val` |
| loss-landscape geometry | `scripts/render_landscape_composite.py` |

Output goes to `figures/`.

## Formal verification

`formal/` is a Lean 4 development that machine-checks the paper's theoretical results against
`mathlib`. Every theorem is kernel-verified with **no `sorry`**: `#print axioms` on each result
lists only Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`). Nothing
domain-specific is axiomatized — where an analytic step is not in `mathlib` it is either built
here or appears as an explicit hypothesis, never as an axiom.

```bash
cd formal
lake exe cache get     # prebuilt mathlib oleans for the pinned toolchain
lake build
```

`formal/README.md` maps each paper result to the file that proves it, and states precisely what is
and is not covered.

## Tests

```bash
pytest tests/ -v
```

Fast, CPU-only, no downloads: the lift emits positive weights from a permutation-invariant summary
and leaves its target frozen and convex, the constrained-weight detection refuses a module it
cannot find weights on, the Karhunen-Loeve basis is linear in its coefficients, the data registry
resolves through `projorg` without opening a socket, and every module and script in the release
imports or parses.

## License

MIT — see [LICENSE](LICENSE). The datasets carry their own licenses; see [Data](#data).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
