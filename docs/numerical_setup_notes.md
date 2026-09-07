# Where each statement in `numerical_setup.tex` comes from

Written 2026-09-07 against HLLD `43bc6c2` and rmhd_final `a2451c7`. The point
of this file is that the section can be re-checked against the code later, and
that anything the code does NOT do is listed rather than quietly written in.

## Evolution code

| statement | source |
|---|---|
| evolved variables `D, Sx, Sy, Sz, tau, Bz`; in-plane B on faces | `src/physics/state.py:41` (`EVOLVED_KEYS`), `State2D.Bxf/Byf` |
| ideal gas, gamma 5/3 rotor, 4/3 Orszag-Tang | `scripts/run_2d.py:28` (`PROBLEMS`), `src/physics/eos.py` |
| piecewise-linear reconstruction, MC limiter (minmod, PCM available) | `src/physics/reconstruction.py:47` (`_LIMITERS`), `:73` |
| MC formula, slopes from the two one-sided differences | `src/physics/reconstruction.py:36-45` (`mc_limiter`), `:59-70` (`cell_slopes`) |
| interface states `Q +- sigma/2`, face i between cells i-1 and i | `src/physics/reconstruction.py:73-95` |
| variables reconstructed: rho, p, vx, vy, vz, Bx, By, Bz | `src/physics/reconstruction.py:100` (`_RECON_KEYS`) |
| reconstruct `z = W v`, recover `v = z / sqrt(1+z^2)` | `src/physics/reconstruction.py:113-125, 160-165` |
| eps recomputed from reconstructed (p, rho) | `src/physics/reconstruction.py:168` |
| per-face PCM fallback on the rho/p floors | `src/physics/reconstruction.py:146-158` |
| two ghost zones | `src/physics/grid.py:166` ("PPM or WENO would need ng = 3") |
| reconstruct `W v` not `v`; PCM floor on rho and p | `src/physics/reconstruction.py:103-151` |
| four flux functions selectable | `scripts/run_2d.py` `SOLVERS`, `src/physics/hlld.py` |
| SSP-RK3 (Shu-Osher), RK2 optional | `src/physics/driver2d.py:100-107` |
| CFL 0.25, unsplit | `scripts/run_2d.py:64`, `src/physics/driver2d.py:52` |
| Kastaun C2P | `src/physics/c2p.py:4,34` (Kastaun et al. 2021 / arXiv:2312.11358) |
| CT with upwinded corner EMF | `src/physics/ct.py:2` (Gardiner & Stone 2005) |
| identical RK weights for cell- and face-centred variables | `src/physics/state.py` `combine()` docstring |

## Exact solver

| statement | source |
|---|---|
| Giacomazzo & Rezzolla 2006 formulation, seven waves, zones R1..R8 | `fullcontact.py:1-25` |
| classic four unknowns + contact closure | `fullcontact.py:17-25` |
| six-unknown vector, log-polar | `fullcontact.py:1115-1131`, `unk6_from_primitive` |
| Alfven wave solved psi-pinned, branch owned by the outer Newton | `fullcontact.py:1119-1121` |
| six residuals: four contact jumps + two slow-wave slacks | `fullcontact.py:1172-1177` |
| slack = eq. (4.17) z-momentum residual (shock) / angle mismatch (rarefaction) | `fullcontact.py:1066-1091` |
| shock root by pressure parameterisation with a bracketing scan | `shock_speed.find_shock_speed_pmethod`, `batched/shock_b.py` scan |
| rarefaction by RK4 through the fan | `rarefaction.py`, `batched/rarefaction_b.py` |
| degenerate families -> reduced three-wave solver; classification from eigenvalues | `batched/classify.py`, `batched/contact_b.py` |
| forward-difference Jacobian, SVD step with truncation, trust region, line search, best iterate | `fullcontact.py:1220-1223`, `batched/fullcontact_b.py` |
| convergence at 1e-8 | `exact_flux.exact_flux_batched(accuracy=1e-8)` |
| MLP: 13 features -> 9 outputs, angles as (cos, sin) | `ml_guess.py:80-105`, `_features_canonical` at `:353` |
| retries from other checkpoints or perturbations | `batched/ml_b.solve_with_retries`, `exact_flux._EXTRA_CKPTS` |
| batched per sweep; exact flux only if converged AND ray resolved AND physical | `exact_flux.py` `take = conv & ray_ok & physical` |
| weak-jump gate | `exact_flux.py` `tau_weak` |

## Rotor

`src/physics/initial_data2d.py:88-119`: r0 = 0.1, rho_in = 10, rho_out = 1,
omega = 9.95, p = 1, B = (1,0,0), sharp edge. Box and end time from
`scripts/run_2d.py:29`. Symmetry diagnostic: `scripts/run_2d.py:36`.

## Deliberately NOT written into the section

- **WENO5.** RESOLVED 2026-09-07: the user confirms it belongs to a different
  code, and it is out of this section. This code has no WENO. Reconstruction
  is piecewise-linear with the MC limiter (minmod and PCM selectable), and the
  grid carries two ghost zones, which WENO5 could not use.
- **HLLC.** Present in the code and listed as selectable, but not used for any
  result in this work so far.
- Numbers that belong to Results, not Setup: the exact fraction per sweep, the
  L1 differences between resolutions, timings.
