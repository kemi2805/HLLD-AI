# Measured interface envelope of the magnetic rotor

**This is the reference for the ML training ranges. Do not guess them again.**

Source: `results/rotor_256_hlld` — a plain-HLLD 256² rotor to t = 0.4
(1321 SSP-RK3 steps, 8407 s, γ = 5/3, CFL 0.25, MC limiter, Gardiner–Stone
constrained transport). Raw samples: `data/rotor_envelope_256.npz`
(795,000 sampled interface states, recorded by `src/physics/envelope.py`).

Reproduce with:

```bash
python scripts/run_2d.py --problem rotor --n 256 --tend 0.4 --solver hlld \
    --out results/rotor_256_hlld \
    --envelope data/rotor_envelope_256.npz --envelope-every 30 --envelope-samples 3000
```

These are the states the Riemann solver is actually handed, in the solver's
own normal/transverse decomposition — the same decomposition the network's
features use.

## The envelope

| field | min | 0.1% | median | 99.9% | max |
|---|---|---|---|---|---|
| `log10 rho_L` | −0.536 | −0.413 | 0.000 | 1.031 | 1.180 |
| `log10 p_L`   | −2.273 | −2.225 | 0.000 | 0.658 | 0.915 |
| `W_L`         | 1.000 | 1.000 | 1.000 | 3.192 | 7.963 |
| `\|B_t\|_L`   | 0.000 | 0.000 | 0.000 | 2.928 | 3.972 |
| `log10 rho_R` | −0.536 | −0.414 | 0.000 | 1.031 | 1.180 |
| `log10 p_R`   | −2.282 | −2.225 | 0.000 | 0.658 | 0.916 |
| `W_R`         | 1.000 | 1.000 | 1.000 | 3.196 | 8.364 |
| `\|B_t\|_R`   | 0.000 | 0.000 | 0.000 | 2.929 | 3.979 |
| `\|B_n\|`     | 0.000 | 0.015 | 1.000 | 3.438 | 5.093 |
| `cos∠(v_t,B_t)` | −1.000 | −1.000 | 0.000 | 1.000 | 1.000 |
| `sigma = b²/(ρh)` | 0.000 | 0.004 | 0.286 | 4.315 | 7.847 |

## How far the current training configuration misses

`configs/default-mignone.yaml` was sampled for the 1D shock tubes. Against
the measured rotor:

| quantity | trained on | rotor needs | |
|---|---|---|---|
| `W` | `[1.0, 2.0]` | up to **8.36** | 4× outside |
| `log10 p` | `[−1, 0]`  (`press: [0.1, 1.0]`, sampled **linearly** despite the name) | `[−2.28, 0.92]` | outside at **both** ends |
| `\|B_n\|` | `[0.01, 1.01]` | up to **5.09** | 5× outside |
| `\|B_t\|` | `[0.0, 1.0]` | up to **3.98** | 4× outside |
| `log10 rho` | `[−1.0, 1.0]` | `[−0.54, 1.18]` | slightly outside at the top |

So the network currently extrapolates over most of the rotor.

## Recommended sampling ranges for the retrain

Measured envelope with ~50% margin. Do **not** restore the wide
commented-out ranges in `default-mignone.yaml`: at a fixed sample count,
wider ranges mean lower density everywhere, and this model is rotor-only.

```yaml
ranges:
  lrho:    [-1.0,  1.5]     # measured [-0.54, 1.18]
  lpress:  [-3.0,  1.4]     # measured [-2.28, 0.92]; NOTE the key rename --
                            # `press` is sampled linearly in generate.py
  Wm1_log: [-4.0,  1.0]     # W = 1 + 10^u  ->  W in [1.0001, 11]
  Bn:      [ 0.0,  7.5]     # measured max 5.09
  Bt_mag:  [ 0.0,  6.0]     # measured max 3.98
```

**Sample `W` as `1 + 10^u`, not uniformly.** The median W is 1.000 and the
99.9th percentile is 3.19, but the maximum is 8.36 — the distribution is
extremely concentrated near 1 with a long tail. Uniform sampling on
`[1, 11]` would put ~90% of the training data in a regime that occupies far
less than 1% of the domain.

## Two findings that change design decisions

**1. The transverse v–B angle is broadly populated.** `cos∠(v_t, B_t)` spans
the full `[−1, 1]` with median 0. The 15-feature vector carries only the
magnitudes `|v_t|` and `|B_t|`, not their relative orientation, so two
states with identical features can have different `p*`. On this problem that
is a **real irreducible error floor, not a theoretical concern** — which
promotes the complete-rotational-invariant feature set from "optional" to
probably necessary. See `src/physics/ai_features.py`.

**2. Near-degenerate normal field persists for the whole run.** `|B_n|` has
its 0.1st percentile at 0.015 and a minimum of 0. The rotor starts fully
degenerate (`B = (B0, 0, 0)` means `By ≡ 0`, so every y-sweep has `B_n = 0`),
but this shows the degenerate and near-degenerate classes keep appearing long
after the field winds up. The reduced three-wave path is needed throughout,
not merely at t = 0.

## Run health, for reference

| | |
|---|---|
| div B | 1.0e-11 for the whole run (machine precision; CT working) |
| π-rotation symmetry | breaks at step ~40 to ~3e-2 — a solver-fallback artefact, not CT (see the plan's trap 3) |
| c2p non-convergences | ~4000 per step out of 65k cells (~6%) |
| HLLE fallback | 0.0–0.2% (x), 90% → 2% (y) as the field winds up |
| rho range | 1.0 → 13.9 peak → 9.4 at t = 0.4 |
