# Warm-start networks across problems (2026-11-03)

Each network seeds the exact solver once on each problem's held-out TEST
faces. The numbers are the faces converged with that one attempt, in %, as
natural / complex.

| network ↓, TEST faces → | rotor (γ 5/3) | OT (γ 4/3) | 4Q (γ 5/3) |
|---|---|---|---|
| production rotor_ft | 94.2 / 78.4 | 97.0 / 86.2 | 87.8 / 90.5 |
| rotor net (validation pick) | **96.6 / 85.5** | 97.8 / 87.0 | 85.6 / 20.4 |
| rotor p_pert (post hoc) | **97.4 / 88.3** | 98.4 / 89.5 | 80.9 / 28.4 |
| OT net (validation pick) | 95.3 / 76.5 | **98.5 / 94.4** | 69.0 / 77.3 |
| OT p_pert (post hoc) | 94.8 / 75.0 | **98.9 / 95.0** | 83.2 / 59.2 |
| 4Q proper net | 85.1 / 47.0 | 95.2 / 76.9 | **91.9 / 97.4** |
| neutral guess | 97.1 / 30.8 | 96.6 / 41.6 | 90.6 / 10.2 |

Bold = the network trained on that problem.

Mean Newton iterations of the converged faces:

| network ↓, TEST faces → | rotor | OT | 4Q |
|---|---|---|---|
| production rotor_ft | 4.87 | 4.57 | 4.79 |
| rotor net (validation pick) | 3.74 | 4.40 | 4.79 |
| rotor p_pert | 3.43 | 4.06 | 5.39 |
| OT net (validation pick) | 4.18 | 3.29 | 6.21 |
| OT p_pert | 4.46 | 3.29 | 5.94 |
| 4Q proper net | 5.36 | 4.95 | 3.35 |
| neutral guess | 4.25 | 4.90 | 4.45 |

## What the numbers are

- **Faces.**
  - **Data:** the 64² exact runs with the current solver (Alfvén closed form
    and slow-shock retry on):
    - rotor and OT: `rotor_64_exact_ssretry` and `ot_64_exact_ssretry`;
    - 4Q: the proper split of job 77245.
  - **Deduplication:** one row per Riemann problem, to ~1e-10. Only
    verified seven-wave solutions are kept.
  - **TEST:** two time blocks, t/t_end = 0.4375 and 0.8125 (± 0.03125).
    Buffers of 0.015625 separate them from the training data. VAL is the
    block at 0.625.
- **Natural** = 6000 TEST faces drawn at random.
- **Complex** = faces whose Alfvén waves rotate the tangential field by
  more than 0.5 rad in total. That is all 1869 for the rotor, and 2000
  drawn for OT and 4Q.
- **The same faces for every network** (fixed draw, rng 20261102). Each
  network gets one Newton attempt, accuracy 1e-8, max_iter 40. This is
  not the solved fraction of a run, which adds retries and rescues.
- **Validation pick** = the checkpoint chosen on VAL only, by the rule
  fixed in advance: most natural faces converged, then complex.
  - **The p_pert rows are post hoc.** They were added after the rotor's
    TEST showed p_pert ahead of the pick in both time blocks.
  - On VAL, all four checkpoints of each problem were within noise of
    each other.
- **Gamma is not a network input.** The features are the two states and
  B_n, one model per EOS. The OT nets were trained at γ 4/3 and the others
  at 5/3, so every off-γ cell runs with a γ mismatch.

## Paired McNemar p-values (natural / complex)

| TEST | comparison | p |
|---|---|---|
| rotor | rotor net vs production | 1.3e-25 / 2e-17 |
| rotor | rotor p_pert vs production | 1.5e-44 / 5.9e-35 |
| rotor | rotor p_pert vs rotor net | 9.7e-6 / 1.6e-6 |
| OT | OT net vs production | 2e-18 / 2.3e-31 |
| OT | OT p_pert vs production | 2e-26 / 1.6e-36 |
| OT | OT net vs neutral | 3.7e-18 / 1.6e-296 |
| OT | rotor net vs production | 9.8e-6 / 0.4 |
| OT | rotor p_pert vs production | 1.3e-14 / 1.1e-5 |
| 4Q | 4Q proper vs production | 4.4e-56 / 5.2e-33 |
| 4Q | rotor net vs production (worse) | 5.9e-5 / ~0 |
| rotor | production vs neutral (neutral better on natural) | 1.2e-21 / 1.2e-196 |
| rotor | rotor net vs neutral (natural a tie) | 0.099 / 2.9e-261 |

## Networks (calea, /mnt/rafast/miler/codes/rotor2d/)

- production: `rmhd_final/data/ml_guess_rotor_ft.pt`
- rotor net: `HLLD/results/ml_retrain/rotor/ml_guess_rotor_p_run_proxy.pt`
- rotor p_pert: `HLLD/results/ml_retrain/rotor/ml_guess_rotor_p_run_pert.pt`
- OT net: `HLLD/results/ml_retrain/ot/ml_guess_ot_p_run_proxy.pt`
- OT p_pert: `HLLD/results/ml_retrain/ot/ml_guess_ot_p_run_pert.pt`
- 4Q proper: `HLLD/results/r2d_retrain/proper/ml_guess_r2d_p_run_pert.pt`

## Jobs and logs

- calea 78178 (rotor), 78179 (OT), 78249 (cross), 77245 (4Q proper).
- Logs: `HLLD/logs/slurm_ml-proper-{rotor_78178,ot_78179,cross_78249}.out`.
- Raw results: `HLLD/results/ml_retrain/cross_{rotor,ot,q4}.{npz,log}`.
