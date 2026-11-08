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
| CIFAR-10 classification | the CIFAR-10 training and test splits | fetched by `torchvision` from the release of Krizhevsky (2009) |

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

## Additional experiments

The code behind the results added to the paper since its first version. Each item runs like the
experiments above, and the commands below are the ones the reported numbers come from. Run them from the
repository root.

**CIFAR-10 classification on the task of Sivaprasad et al. (2021)**, Appendix E. Their
input-output-convex classifier, three hidden layers of 800 units with ELU activations, is trained
on one pipeline under six positivity methods: their rule, which replaces each negative constrained
weight `w` by `exp(w - 5)` after every step (`exppost`), the same rule with `exp(w - 8.8)`, whose
return point is the median of the initial weights (`exppost_matched`), direct softplus (`direct`),
PGD (`pgd`), the lift (`lift`), and the same network without positivity (`nonconvex`). The driver is
`scripts/experiments_sivaprasad_cifar.py` with `configs/experiments_sivaprasad_cifar.json`; the
step size of each method is selected on validation accuracy, and test accuracy is read at the epoch
of best validation accuracy.

```bash
# preflight, once per step size: checks that every setting takes effect and trains nothing
for LR in 0.0001 0.0003 0.001; do
    python scripts/experiments_sivaprasad_cifar.py --phase preflight --lr $LR --seeds 0
done
# step-size selection: seed 0, every method at every step size
for LR in 0.0001 0.0003 0.001; do
    for C in lift pgd direct exppost exppost_matched nonconvex; do
        python scripts/experiments_sivaprasad_cifar.py --phase cell --constructions $C --lr $LR \
            --seeds 0 --checkpoint_every_epochs 5
    done
done
python scripts/collect_sivaprasad_cifar.py --select_only --select_seeds 0
# seeds 1 to 4, each method at its selected step size
for C in lift pgd direct exppost exppost_matched nonconvex; do
    LR=$(python -c "import json; print(json.load(open('data/analysis/sivaprasad_cifar_selection.json'))['selected']['$C']['lr'])")
    for S in 1 2 3 4; do
        python scripts/experiments_sivaprasad_cifar.py --phase cell --constructions $C --lr $LR \
            --seeds $S --checkpoint_every_epochs 5
    done
done
# the table and curves, then the figure
python scripts/collect_sivaprasad_cifar.py
python scripts/render_sivaprasad_cifar.py
```

A lift cell at this width holds about 2.4e8 parameters; the reported runs used a 16 GB GPU per cell.

**Two further positivity methods on the one-dimensional targets.** The exponential map
`W = exp(W_raw)` (backend `direct_exp`, and `hypernet_exp` for the lift's readout) and the rule of
Sivaprasad et al. (backend `direct_exppost`; `direct_exppost<E>` sets the constant), both through
`scripts/experiments_lcmm_1d_logconcave.py`. `scripts/collect_positivity_maps_1d.py` reads the
total-variation distance at the best held-out loss from the checkpoints.

```bash
python scripts/experiments_lcmm_1d_logconcave.py --targets gumbel,laplace,gamma,beta \
    --seeds 0,1,2 --backends direct_exppost --out_dir data/checkpoints/lcmm_1d_exppost
python scripts/experiments_lcmm_1d_logconcave.py --targets gumbel,laplace,gamma,beta \
    --seeds 0,1,2 --backends direct_exp --out_dir data/checkpoints/lcmm_1d_exp
python scripts/collect_positivity_maps_1d.py --out_dir data/checkpoints/lcmm_1d_exppost \
    --backends direct_exppost
python scripts/collect_positivity_maps_1d.py --out_dir data/checkpoints/lcmm_1d_exp \
    --backends direct_exp
```

**Shoulder share and exit rate at three thresholds**, Table 5. `scripts/measure_occupancy_thresholds.py`
recomputes the statistics of `scripts/measure_dwell_time.py` from the snapshots stored in the
thirty-seed one-dimensional archives, at the thresholds 0.02, 0.05 and 0.10 on the derivative of the
map and on a grid of weight edges, with no training. It checks first that it reproduces
`measure_dwell_time.py` at the paper's threshold.

```bash
python scripts/measure_dwell_time.py
python scripts/measure_occupancy_thresholds.py
```

**Pinned evaluation under any backend.** `scripts/reeval_ebm_pinned.py --backend` reads the lift
(`hypernet`, `hypernet_exp`) or direct training (`direct`, `direct_exp`). The positivity map
(`--hyper_pos_mode`) and the summary standardization (`--pool_norm`) hold no parameters, so a
replay guard rebuilds every stored emission from its saved state and batch and refuses a mismatch;
`--guard_only 1` runs the guard alone. `--nll_hist 1` adds the held-out NLL at every snapshot on one
fixed sample.

```bash
python scripts/reeval_ebm_pinned.py --archive data/checkpoints/lcmm_1d_exp/gumbel \
    --backend direct_exp --target_kind gumbel --config configs/experiments_lcmm_1d_logconcave.json \
    --snapshots 1 --nll_hist 1 --out data/analysis/pinned/lcmm_1d_exp_gumbel
```

**The lift under the exponential map, with the step size selected.** Direct training under the exponential map (`direct_exp`), the lift under it
(`hypernet_exp`) and the lift under softplus (`hypernet`) are trained on the four one-dimensional
targets through `scripts/experiments_lcmm_1d_logconcave.py`. The step size of each method on each
target is selected from 5e-4, 1e-3 and 2e-3 on the best held-out loss of seed 0, and seeds 1 to 5
are trained at the selected step. Every method is then re-evaluated with the replay guard, the lifts
with their emission pinned, and the held-out NLL of every run is read at each snapshot on one fixed
sample.
`scripts/collect_exp_map_1d.py` makes the selection and the table of median total-variation
distances at the best held-out loss. It also reports the seeds on which the lift under the
exponential map is below direct training under it, and the first iteration at which each of its
runs reaches the best held-out loss of any direct run.

```bash
# step-size selection: seed 0, every method at every step size
for LR in 5e-4 1e-3 2e-3; do
    python scripts/experiments_lcmm_1d_logconcave.py --targets gumbel,laplace,gamma,beta \
        --backends direct_exp,hypernet_exp,hypernet --hyper_pool_norm 1 --seeds 0 --lr_ebm $LR \
        --out_dir data/checkpoints/exp_map_1d/A_lr$LR
done
python scripts/collect_exp_map_1d.py --select_only
# seeds 1 to 5, each method at its selected step size, then the pinned re-evaluation
for M in direct_exp hypernet_exp hypernet; do
    for T in gumbel laplace gamma beta; do
        LR=$(python -c "import json; print(json.load(open('data/analysis/exp_map_1d_selection.json'))['$M']['$T']['lr'])")
        python scripts/experiments_lcmm_1d_logconcave.py --targets $T --backends $M \
            --hyper_pool_norm 1 --seeds 1,2,3,4,5 --lr_ebm $LR \
            --out_dir data/checkpoints/exp_map_1d/B_${M}_lr$LR
        python scripts/reeval_ebm_pinned.py --archive data/checkpoints/exp_map_1d/B_${M}_lr$LR/$T \
            --backend $M --pool_norm 1 --target_kind $T \
            --config configs/experiments_lcmm_1d_logconcave.json --snapshots 1 --nll_hist 1 \
            --seeds 1 2 3 4 5 --out data/analysis/pinned/exp_map_1d/B_${M}_$T
    done
done
# the table
python scripts/collect_exp_map_1d.py
```

The 96 runs took about five hours on one 16 GB GPU, and each lift checkpoint takes 240 MB.

**Revised figures.**

| figure | command |
|---|---|
| dwell on the shoulder | `python scripts/render_dwell_survival.py` |
| image regularizer reconstructions | `python scripts/render_ffhq_panel.py --curve none` |
| image regularizer dynamics | `python scripts/render_ffhq_resilience.py --panel_c` |
| image regularizer across step sizes | `python scripts/render_ffhq_resilience.py --sweep_only` |
| CIFAR-10 classification | `python scripts/render_sivaprasad_cifar.py` |
| loss-landscape geometry | `python scripts/render_landscape_composite.py` |

**Captioned figures.** `scripts/render_additional_figures.py` redraws seven captioned figures of these
results from the records the commands above produce, into `figures/additional/`.

```bash
python scripts/render_additional_figures.py --figure all
```

**Theory.** The stationary law of the projection and Spitzer's identity are now proved, in five new
modules of `formal/` (`SpitzerCyclic`, `SpitzerIdentity`, `SpitzerGaussian`, `LindleyStationary`,
`LindleyClamped`); `formal/README.md` maps each to the statement it backs.

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
resolves through `projorg` without opening a socket, the exponential map and the post-update rule
act as stated, the step-size selection on the one-dimensional targets follows its rule, the
CIFAR-10 constructions apply their maps, pass their step-zero checks and resume exactly, and every
module and script in the release imports or parses.

## License

MIT — see [LICENSE](LICENSE). The datasets carry their own licenses; see [Data](#data).

## Contact

Ali Siahkoohi — <alisk@ucf.edu>
