# Where each statement in `planar_solver.tex` comes from

Written 2026-09-08 against HLLD `954046b` and rmhd_final `add9f70`. Same
purpose as `numerical_setup_notes.md`: the section quotes measurements, so
each one needs a source that outlives the session.

**Everything marked MEASURED is re-derivable by one command:**

```
python scripts/measure_planar_claims.py results/rotor_64_exact/harvest \
    --dataset /Users/miler/Codes/rmhd_final/data/dataset_rotor_train.npz
```

Its numbered blocks match the order of the section's claims. The published
numbers come from the 64² run of 2026-09-08 (284 steps to t = 0.4, 481,837
solved interfaces). Numbers that need a run rather than a harvest are noted
individually below.

## Sect. 1 — why the seven-wave form fails

| statement | source |
|---|---|
| coplanarity median 3.5e-11, below 1e-6 on 100% | MEASURED, block 1 |
| forward-constructed set: median 0.65, 0.01% below 1e-4 | MEASURED, block 1, `--dataset` |
| slow wave turns the field by 6e-12 rad | MEASURED, block 2 |
| freezing the angles: 79.5% → 90.5% | session measurement 2026-09-08 on `unsolved`/`solved` shards; the mechanism is asserted by `batched/test_planar5.py` rather than the number. Recorded in [[rotor-failures-no-root]] |
| slow wave changes \|Bt\| by 3.3%, under 1% on 27.5% | MEASURED, block 3 |
| ~50% constructible from a seed, 99.5% in a grid | session measurement (angle enumeration × 7³ grid); [[rotor-failures-no-root]] |
| Alfven/slow gap median 5.5e-3, p5 9.4e-6; fast/Alfven 2.8e-1 | MEASURED, block 4 |
| exhaustive 8 × 7³ = 2744 points converges 8.5%; 3³ gives 7.0% | session measurement; [[rotor-failures-no-root]] |

The three ill-posedness claims are also stated, with the same numbers, in the
module docstring of `rmhd_final/batched/planar5_b.py`.

## Sect. 2 — the reduced system

| statement | source |
|---|---|
| the five-wave formulation, unknowns and residual | `rmhd_final/batched/planar5_b.py`, `structure()` |
| slack is an identity, ~1e-11 | `batched/test_planar5.py::test_slow_wave_slack_is_an_identity`; MEASURED, block 6 |
| signed `Bt_CD` removes the discrete branch | `planar5_b.structure` passes `Bz_t = 0` to `slow_wave6`; the eight-way enumeration it replaces is in the session's `enum8` measurement |
| no warm start needed | `batched/test_planar5.py::test_no_warm_start_is_needed`; every number in the section comes from the default seed in `planar5_b.solve` |
| planar frame, out-of-plane residue reported | `planar5_b.to_planar`; residue median 1.5e-10, 0.2% above tolerance — MEASURED, block 6 |

## Sect. 3 — verification

| statement | source |
|---|---|
| the answer is written in the seven-wave unknowns and checked | `planar5_b.verify_full`, called by `solve` before setting `converged` |
| accepted only below 1e-8; median 1.1e-10 | `RMHD_PLANAR5_VERIFY` default in `src/physics/exact_flux.py`; MEASURED, block 6 |
| pi rotations: 2.3% affected, 97.7% have both zero | MEASURED, block 5 |
| a pi rotation is outside the family and must be rejected | `batched/test_planar5.py::test_pi_rotations_are_rejected_not_faked` |
| three-wave solver: converges ~100%, never reaches 1e-6, median 1.2e-2 | MEASURED, block 7 |
| the residual is frame-sensitive | MEASURED, block 8 |

**The frame claim is deliberately weak, and the reason is in the data.** Two
harvests disagree on the *direction* for seven-wave rows: on the 64² harvest
those rows scored better in the raw frame, on the pilot harvest better in the
planar frame. What both agree on, and what the section states, is that the
numbers move by tens of percent, and that rows produced by the planar solver
score 100% in the planar frame against 51% in the raw one with a tenth
unconstructible. Hence: verify in the frame the answer was made in. Do not
strengthen this without a third, larger measurement. See
[[sevenwave-frame-dependent]], which is correspondingly hedged.

## Sect. 4 — measured behaviour

| statement | source |
|---|---|
| the comparison table | MEASURED, blocks 6 and 7, except the cost row |
| cost ~8 ms (planar), ~1 ms (three-wave), ~30 ms (seven-wave) | session timings; the run logs give the per-step figure |
| replay 471 → 763, 292 gained, none lost, p = 2.5e-88 | `scripts/rotor_replay.py replay x nj800x3` with `RMHD_PLANAR5_FALLBACK` off and on, then `compare`; recorded in the commit message of `2bfe9aa` |
| p* bitwise identical on lanes both resolve | same, session check over the saved replay masks |
| coverage 43.9% → 69.9%; y-sweeps 24.7% → 83.9% | `scripts/harvest_summary.py` on `results/rotor_64_exact/harvest` and on the run with the rescue enabled |
| factor ~2 in wall time per step | the two run logs (`rotor_64_exact.log`, `rotor_64_p5.log`) |

## Deliberately NOT claimed

- **That the planar frame is better for solving.** It is only used for
  checking. Canonicalising the frame before the seven-wave Newton is a
  plausible improvement and is untested; nothing has been changed on that
  basis.
- **That the planar solver should be the primary path.** It needs no warm
  start and is well conditioned, so this is worth measuring, but it would
  change every run's answer.
- **Any claim that the exact flux improves the solution.** It does not, at the
  coverage measured so far: the flux changed the 64² solution by 0.37% and the
  128² solution by 0.82% while the resolution gaps are 33% and 27%. Whether
  ~80% coverage changes that is the open question the rescue was built to
  answer, and the run is in progress.
