# Fixing the four-quadrant 128² losses, 2026-11-04

User: "I am very interested in what would fix it. Please try them out and lets
see if we can get to 100%".

Population: the failure ledger of the 128² ft3 run (calea 78407):
- 872 truly lost lanes: type A 423, B weak 328, B strong 121;
- 1,228 controls.

Paired replays run the same 2,100 faces through production twice, with the switches off and on (`replay_pop.py`, `compare_replay.py`).

## What the losses actually were

**Type A (the 0.99c shear faces) is not a compound wave.** Lane 62 is two ordinary slow shocks, with |B_t| falling toward the contact from both sides. The earlier reading confused UL's left slow and right Alfvén speeds with "right slow / Alfvén".
- The cause is a solver defect. In a band of contact fields around the root, the left slow-shock ladder converges from no seed and returns its **loose fallback**: a grid cell with Vs pinned at a grid node and P_tot 14% low, flagged as a success (`probe_hole.py`).
- So the planar residual has a hole exactly at its zero, and the slow-shock retry, which only runs on failed lanes, never sees it.
- The field-flip branches converge instead, but production rejects them as self-crossing (403 of 423 lanes, `ledger_why.py`).

**Type B weak (near-normal field) is not intermediate shocks.** Lane 215:
- No π-rotation exists for these relativistic states: `exact_rotation` has no root for dphi in about (0.3, 2.8).
- Every reversing slow "shock" from them is an expansion shock (entropy falls, Lax fails; `probe_is.py`).
- The solution is a slow **rarefaction that carries B_t through zero** on each side. It is narrower than one tube cell (2e-4 in xi); the tube smears it.
- A fan stepped in |B_t| cannot cross zero; a pressure-stepped one can, and is isentropic to 1e-14 (`probe_pfan.py`).
- Planar4 misses it for two reasons:
  - it steers by field;
  - its mean-pressure start is 15% off, because both fast waves are rarefactions.

**The near-degenerate Newton.** On lane 1099 the four-unknown Newton zig-zags at 0.9 per iteration: its forward-difference Jacobian is too inaccurate at condition ~2e4. Central differences converge quadratically (`probe_p4cd.py`).

**β ~ 100-300, field normal to 0.2%.** The slow wave is nearly an Alfvén wave: a total-pressure change of 1e-5 swings B_t by 0.1 (lane 900).
- Pressure is then an ill-conditioned coordinate.
- |B_t| cannot cross zero.
- The **signed B_t** is the right coordinate: kind S (`fan_signed`, the pressure fan's ODE re-parametrised by B_t).

## The fixes (all off by default, bitwise the old code when off)

| switch | where | what |
|---|---|---|
| `RMHD_SLOWSHOCK_LOOSE_RETRY=1` | rmhd slow_shock_b + njit | lanes the ladder answered with its loose fallback go to the retry; a lane the retry cannot answer keeps its cell |
| `RMHD_PLANAR4_STAR=1` | HLLD exact_flux `_planar4_star` + rmhd planar4 `unk_init`, `central` | planar4's lost lanes again from HLLD's p*: starts PP/PB/BP; forward differences, then central; same acceptance, and the entropy may not fall across any shock |
| kind S (in progress) | rmhd planar4 `KIND_S` | the slow wave to a signed B_t: `fan_signed`, or the Bt-method slow shock |

Worktrees:
- local: `rmhd_fix` (branch fix-loose-retry) and `hlld_fix` (branch fix-planar4-star at 51e9c67);
- calea, frozen for the 64² runs: `rmhd_final-loose` and `HLLD-fix` (patches `run64_patches/`);
- calea, for new experiments: `rmhd_final-loose2` and `HLLD-fix2`.

Tests:
- `tests/batched/test_slow_shock_loose_retry.py` (6 tests);
- `tests/batched/test_planar4_start.py` (4 tests).

## Results (paired replays on calea, controls bitwise every time)

| replay | A (423) | B weak (328) | B strong (121) | all (872) |
|---|---|---|---|---|
| loose retry (78575) | 400 | 0 | 0 | 400 |
| + planar4 from p* (78597) | 404 | 120 | 66 | 590 |
| + central differences (78601) | 404 | 151 | 68 | **623 (71%)** |

**Against the lanes' own 1D tubes:** HLLD vs tube is 2e-3..1.5e-2; the exact flux is
- 3e-5..3.6e-4 on type A, falling with resolution;
- 5e-7..8e-7 on type B (lanes 215, 1069, 1348);
- with kind S, 1e-8 / 2e-7 / 1.8e-4 (lanes 900, 1245, 214), against HLLD 2.5e-5 / 1.2e-5 / 8.8e-3.

Kind-S answers can contain field-reversing jumps whose entropy falls slightly (1245: −1.4e-7; 214: −2.2e-5). They still beat HLLD against the tube by 50-60x. Whether to keep them in the run is the user's call (cf. the inadmissible slow shocks: dropped from training, kept in the run when better than HLLD).

## Cost (peer report 11-05, early_cmp.py)

The 64² fix runs are 1.85-2.9x slower per step than their controls (4Q only from step 100 on). The cause is the star retry, not the loose retry:
- loose retry alone on the ledger replay: +3% wall;
- star retry, forward starts: +45%;
- star retry, forward + central: +80%.

It runs up to 10 starts x 40 numpy Newton iterations on every lost lane, and that per-call overhead does not spread over the pool. Lean set: forward PP; central PP, SS, BP, PB; at most 25 iterations (RMHD_PLANAR4_STAR_SET, _ITER). The signed fan is compiled (fan_njit.make_signed, numpy == njit to 10 digits). Replays 78625 (full) and 78626 (lean) measure both.

## B strong (the last ~52)

Lane 858 has a regular structure: LF shock, slow fans, RF shock 6e-4 from the Alfven speed.
- It converges from the TUBE plateaus with every steering.
- It does not converge from p*: the Newton stalls at 2e-3 with cond 2e5-9e5.
- It does not converge from a jump homotopy either, which fails already at theta = 0.1.
- The cause: p* is 1.7% from p_RF, and near the fast-Alfven degeneracy the basin in p_RF is narrow.

Fix: start a weak fast wave (p* within 2% of its ahead pressure) at its ahead pressure x (1 +- 1e-3). From that start lane 858 converges with every steering (`START=side`). Experiment 78627 runs it on the lanes kind S still loses.

## Running

- 64² paired runs, both switches on, controls as for adm64: 78603 (4Q), 78604 (OT), 78605 (rotor). All three pilots match their controls bitwise.
- 78609: kind S plus the old steerings on the 249 lanes still lost.
- 78606: tubes of 16 of those lanes (`tubes_rem/`).

## FINAL configuration (2026-11-05)

`RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4`, lean starts. The star retry on the lanes planar4 lost tries, in order:
- forward PP;
- central PP, SS, BP, PB;
- the near-side PP (+-1e-3);
- both fast waves at their own ahead pressures (+-1e-3, +-1e-2).

Each attempt runs at most 25 iterations.

Patches: `final2_patches/`. Calea worktrees, frozen: rmhd_final-loose5, HLLD-fix5.

**Replay 78639 (872 lost faces):**

| entropy may fall by | A (423) | B weak (328) | B strong (121) | all | wall vs off |
|---|---|---|---|---|---|
| 1e-9 (strict) | 419 | 214 | 110 | 743 (85.2%) | 1.77x |
| **1e-4** | **422** | **328** | **110** | **860 (98.6%)** | **1.53x** |

- The 1,228 controls are exact in both arms, with bitwise-equal fluxes.
- 12 faces are still lost: 155, 202, 213, 217, 273, 333, 534, 554, 642, 910, 921, 1497.
- The entropy-falling answers (1e-10..2e-5) are 10-300x closer to the faces' own tubes than HLLD, on every one checked (6 lanes).

**Paired 64^2 runs with it:** 78636 (4Q), 78637 (OT), 78638 (rotor), writing `results/*_64_exact_fixfinal2`. All three pilots are bitwise. They supersede the fixfinal (78631-78633) and fix64 OT/rotor (78604-78605) runs, all cancelled. fix64 4Q (78603, the first config) is left to finish.

## Are the answers actual solutions of ideal RMHD? (verification, 2026-11-05)

Tools, all of them in `fixes/`:
- `rmhd_ideal.py`: the conserved variables and fluxes written from the equations. They agree with HLLD's flux code to 1e-13.
- `verify_rmhd.py`: checks every wave of an answer in the harvest schema:
  - every jump against all seven jump conditions at its speed (tolerance 1e-6);
  - every fan re-integrated from its own state ahead (pressure or signed B_t), the self-similar identity F - xi U + int U dxi = const with Simpson's rule (33 samples, 257 where needed), and that its characteristic speed spreads;
  - the wave order (production's tolerance, 1e-6);
  - the entropy across every shock and rotation.
- `capture_answers.py`: every answer on the ledger (harvester, plus wrapped four-unknown calls).
- `sample_run_harvest.py`: 256,000 answers from the 128^2 run's harvest.
- `verify_final.py`: one category per answer.

Calea jobs: 78642 (capture), 78749 (final verification).

| answers | n | valid | entropy falls at a jump | "fan" that compresses | slow wave past the fast one | other |
|---|---|---|---|---|---|---|
| 128^2 run, seven-wave | 234,340 | 99.06% | 0.55% | 0.17% | 0.19% | 0.03% |
| 128^2 run, planar | 21,660 | 95.35% | 1.99% | 1.73% | 0.90% | 0.03% |
| ledger, fixes: planar (type A) | 369 | **100%** | 0 | 0 | 0 | 0 |
| ledger, fixes: four-unknown | 435 | 74.0% | 25.8% | 0 | 0 | 0.2% |
| ledger controls (production) | 1,214 | 98.5-99.7% | | | | |

- **Conservation holds everywhere it is tested:**
  - jumps at the median residual 3e-13;
  - fans on their integral curves to 2e-14;
  - the fan identity converges at fourth order with the number of samples.
- **"Other":**
  - 48 contacts closed to 1-2e-6 (the Newton tolerance);
  - 47 fans really off their integral curve (0.09 median): 0.02% of answers.
- **The verifier's own artefacts, now fixed:**
  - the out-of-plane floor;
  - the 17-point trapezoid;
  - the 1e-13 rotation threshold;
  - an order tolerance stricter than production's 1e-6.


## Admissibility gate, paired (2026-11-05)

User: "yes, build the acceptance check and test it paired".

**What the gate does.** `RMHD_ADMISSIBLE_GATE=1` (HLLD `exact_flux._admissible_structure`, off by default; patch `gate_patches/gate_hlld.diff`, calea worktree HLLD-fix6). Where the seven-wave, planar and four-unknown solvers accept an answer, it is refused if:
- a shock or rotation lowers the entropy by more than `RMHD_ADMISSIBLE_GATE_DS` (1e-9);
- a fan's characteristic speed does not spread;
- a wave sits behind the previous non-zero one (fans as their whole range, slow against fast included; production's 1e-6 tolerance).

A refused answer goes on to the next rung, and to HLLD if none answers it. With the gate off, the code is bitwise the old one.

**Ledger replay 78912** (the 872 lost faces + 1,228 solved faces of the 4Q 128^2 run, four arms on one node; `compare_gate.py`, results in `replay_gate/`):

| arm | lost faces answered | solved faces kept | wall |
|---|---|---|---|
| production | 0 / 872 | 1228 / 1228 | 489 s |
| production + gate | 0 / 872 | 1220 / 1228 | 495 s |
| fixes (entropy may fall 1e-4) | 860 / 872 | 1228 / 1228 | 758 s |
| fixes + gate | 743 / 872 | 1227 / 1228 | 892 s |

What the gate refused:
- **12 of production's 1,228 answers (1.0%):** 8 seven-wave, 4 planar.
  - Production's rescues re-answer 4 of them; the flux moves by a median of 3e-5 (at most 5e-4).
  - With the fixes, the star starts re-answer 11 of the 12.
- **117 of the 860 recovered lost faces:** 108 near-normal, 3 type A, plus 1 control.
  - Every one carries a field-reversing slow jump (LS 62, RS 50) whose entropy falls by 1e-9 to 7.4e-5 (median 4e-7).
  - 66 of them fall by less than 1e-6, which is the accuracy the jump conditions are solved to; 46 fall by 1e-6 to 7.4e-5.

**Estimate for the 4Q 128^2 run** (lost 0.80%):

| | conservative exact answer | admissible as well |
|---|---|---|
| production | 99.20% | ~98.6% |
| fixes | 99.99% | ~99.8% |

**Paired 64^2 runs** (`calea_gate64.sh`, all four pilots bitwise):
- 78915: 4Q, fixes + gate;
- 78916: OT, fixes + gate;
- 78917: rotor, fixes + gate;
- 78918: 4Q, production + gate.

Wall so far against the controls:
- 4Q fixes + gate: 0.86-1.02x;
- 4Q production + gate: 0.97-1.02x;
- OT fixes + gate: 7.2x over steps 1-10 and 2.2x over steps 10-25 (fixes alone: 3.6x / 1.3x / 3.9x over steps 25-50 / 1.9x over steps 50-100);
- rotor fixes + gate: 3.1x / 2.3x (fixes alone: 3.0x / 1.7x / 1.3x).

**Fine tubes (78641, `tube_conv.py`): do the faces' own 1D tubes converge to answers the gate refuses?** Flux at x/t = 0 against the replay's exact (fixes) and HLLD fluxes, max relative difference over the components:

| lane | gate | entropy change at the slow jumps | HLLD | exact, 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| 214 | refuses | -2.2e-5 / +1.1e-5 | 8.7e-3 | 1.8e-4 | 1.6e-5 | 2.6e-5 | 9.0e-6 |
| 788 | refuses | -1.3e-6 / -1.3e-6 | 1.6e-4 | 2.4e-5 | 4.1e-6 | 2.4e-6 | 2.2e-6 |
| 1013 | refuses | -7.5e-8 / -8.6e-8 | 1.3e-4 | 3.2e-6 | 1.1e-6 | 5.4e-7 | 3.8e-7 |
| 1245 | refuses | -1.4e-7 / -1.2e-7 | 1.2e-5 | 3.3e-7 | 2.0e-7 | 3.3e-7 | 1.2e-7 |
| 1299 | refuses | -3.0e-8 | 1.9e-5 | 4.9e-7 | 2.8e-7 | 2.7e-7 | 2.0e-7 |
| 1148 | keeps | +2.8e-7 / +2.0e-7 | 4.2e-4 | 4.0e-6 | 2.8e-6 | 1.8e-6 | 1.1e-6 |
| 1406 | keeps | +1.6e-7 / +1.3e-7 | 3.2e-3 | 1.9e-5 | 1.0e-5 | 6.0e-6 | 3.5e-6 |

- The tubes keep approaching the exact flux up to 16384 cells: they end within 1e-7..9e-6 of it, where HLLD stays 1e-5..9e-3 away.
- Entropy-falling and entropy-rising field-reversing jumps behave the same way, at the same size (1e-7).
- So the refused answers' FLUXES are the fine-grid limit to at least 1e-5. What is inadmissible is only the wave structure, by a margin of the size of the solve's own accuracy.

## The refused answers are a rotation fused into the slow jump (2026-11-05)

User: "take a look on the entropy fall, we may get this also up to 99.99%" and "Are all faces now solved with the seven wave family?"

**What the refused slow jumps are** (`reversing_anatomy.py`, the 162 entropy-falling jumps on 112 refused lanes):
- All but two reverse the tangential field and usually grow it (behind / ahead median -4).
- All expand: the density falls by 5e-6..2e-2, and the characteristics diverge from the jump (none is Lax).
- All but one run between the Alfven speeds of their two sides.
- The faces have a near-normal field at beta ~100-300, so the slow and Alfven speeds agree to ~1e-9.

**Every such side has a rotation; the lab sign decides whether production finds it.** A coplanar Alfven rotation turns the field through pi in the fluid frame. In the lab its B_t lands on either sign:
- lane 1013, right side: +0.057 -> +0.022;
- lane 166, right side: +0.365 -> +0.030.

Production's reversal (planar5 `_reverse`: the pinned solve and `exact_rotation` at the lab angle + pi) finds only the opposite-sign ones. `rd_helpers.coplanar_rotations` (the same closed form at both lab angles, identity excluded) finds exactly one rotation per side, to a jump residual of ~1e-16.

**Re-solve** (`rd_solve.py`, calea 79423):
- Setup: fast wave, then the rotation, then a signed-field slow wave, on each side; four unknowns; Newton from the refused answer; every combination of rotations; judged by `verify_rmhd`.

| refused lanes | admissible answer found | one combination works | two work | rotation on both / left / right | flux change vs refused, median (max) |
|---|---|---|---|---|---|
| 112 | **112** | 108 | 4 | 54 / 38 / 24 | 9e-9 (4.8e-6) |

- Newton converges in 0-2 iterations; the refused answer is already within ~1e-8 of the admissible one.
- Tube lanes (16384 cells): the distance to the tube is unchanged (1013 3.7e-7, 788 2.2e-6, 1245 1.2e-7, 1299 2.0e-7, 214 1.1e-5).

**Production's own answers outside the regular family:** in the 256,000-answer sample of the 4Q 128^2 run, 5,207 answers (2.0%) contain a slow jump that crosses the Alfven speed:
- most cross by 1e-8..1e-4;
- the entropy falls on 1,350 of the 7,563 crossing jumps.

**Built (uncommitted, `rot_patches/`):**
- rmhd: `fullcontact_b.coplanar_rotation`; `planar4_b` `structure` / `jacobian` / `solve` take `rot`; `certify` takes the rotations.
- HLLD: `RMHD_PLANAR4_ROT` (off by default):
  - `_keep_refused` stores the first refused four-unknown answer;
  - `_planar4_rotation` re-solves the lanes still lost from it, combinations (both, left, right), accepted by production's acceptance plus the gate.
- Batched check: `test_p4rot.py`.

**Production's answers across the Alfven speed, re-solved as regular** (`rd_solve_run.py`, calea 79431). These are the 5,207 answers of the 256k 4Q 128^2 sample that contain a slow jump crossing the Alfven speed: fast wave, rotation, signed-field slow wave, Newton from the answer itself, independent verifier.

| original jump's entropy change | answers | regular admissible answer found |
|---|---|---|
| falls (< -1e-9) | 1,207 | 432 (36%) |
| about zero (+-1e-9) | 1,598 | 920 (58%) |
| rises (> 1e-9) | 2,402 | 2,086 (87%) |
| **all** | **5,207** | **3,438 (66%)** |

- Where one is found, the flux moves from the original by a median of 6e-10 (99th percentile 4e-8, max 4.4e-7).
- The rest are:
  - 1,427 regular weak solutions with a jump whose entropy falls;
  - 324 that fail the verifier otherwise;
  - 18 where no combination converges.
- Per-wave detail: calea 79435 (`rd_run_detail/`).

**Why two answers can give the same flux** (`figs_two_answers.py` -> `figs/two_answers_1013.png`). On lane 1013 the refused answer and the regular one differ only inside a sliver ~1.5e-5 wide around each slow/Alfven pair. The four plateaus both answers share (R2, R4, R5, R7) agree to 3e-8..5e-8, and the face flux to 4e-9.

Over the 7,563 crossing jumps of production's answers:
- the density change across the jump has a median of 6.7e-4;
- the slow-Alfven speed gap ahead has a median of 9e-6;
- beta has a median of 120.

The flux change after the regular re-solve (median 1.1e-9) scales as the cube of the density change (median 8e-10; log-correlation 0.61). A jump and a fan of the same small strength reach end states that agree to third order.

**64^2 runs with the rotation re-solve** (user 11-05: "yes, launch the 64² runs with the rotation re-solve"): `calea_rot64.sh` MODE=finalrot, submitted as 79440 (4Q), 79441 (OT) and 79442 (rotor), writing `results/*_64_exact_finalrotgate`. They were queued at submission and their pilots are watched.

**Paired ledger replay with the rotation re-solve** (calea 79433, `calea_replay_rot.sh`, results in `replay_rot/`, all arms with the gate on):

| arm | lost faces answered | solved faces kept | wall |
|---|---|---|---|
| production + gate | 0 / 872 | 1220 / 1228 | 504 s |
| production + gate + rotation | 0 / 872 | 1222 / 1228 | 508 s |
| fixes + gate | 743 / 872 | 1227 / 1228 | 902 s |
| **fixes + gate + rotation** | **860 / 872** | **1228 / 1228** | 910 s |

- The rotation re-solve answers all 117 lost faces and the one solved face the gate had refused.
- It takes no answer away. Every face exact in both arms has a bitwise-equal flux.
- It costs +1% wall on this population.
- The admissible count now equals the count without the gate: 860, with 12 faces left (no answer at all).
- Estimate for the 4Q 128^2 run: **99.99% of faces exact AND admissible** (lost 0.80% x 12/872 = 0.011%).

**The 34% without a regular admissible answer** (calea 79435, `rd_run_detail_report.py`). The rerun reproduces 3,438 / 5,207. Of the rest:
- **1,377 become regular on one side only.**
  - The rotation fixes the left (617) or the right (750) crossing jump. On the other side the slow jump still crosses the Alfven speed and lowers the entropy (median 9e-8; above 1e-6 on 307).
  - A rotation on that side as well fails because the waves run out of order (817 order failures). The slow wave would have to run ahead of the rotation.
  - 560 of these 1,377 were admissible as production gave them (the crossing jump raised the entropy). For those the planar, intermediate answer is the admissible one we have.
- The remaining ~390 fail the verifier otherwise or do not converge.

## The seven-wave solver with signed-field slow waves and the rotation (2026-11-05)

User: "give the seven-wave solver signed-field slow waves and the rotation. be ambitious".

**Why a second formulation.** `fullcontact6` solves for ONE contact field that both slow waves are built towards (unknowns: ln p_LF, ln|Bt_CD|, psi_CD, ln p_RF, phi_L, phi_R). On a coplanar face (every face of a 2D run) that cannot express:
- per-side steering;
- a slow fan through B_t = 0;
- the coplanar rotation that keeps the lab sign (at a pinned lab angle that rotation is the identity).

The four-unknown planar form has all three: the unknowns are two fast-wave pressures and one parameter per slow wave, the field continuity is an equation, and a rotation is a branch.

**Built** (uncommitted, `c7_patches/`):
- `rmhd/batched/coplanar7_b.py` (new) and `api.solve_batch`: with `RMHD_COPLANAR7=1` the COPLANAR class is solved FIRST in the planar form, seeded by the network seeds, in batched stages:
  1. the seed's rotations;
  2. the other three rotation choices;
  3. the extra checkpoints' seeds x all four.
- A lane is answered by the first branch that converges, passes planar4's certificate and raises the entropy at every shock. Unanswered lanes fall back to `fullcontact6`.
- The answer is returned in the seven-wave layout, with `coplanar7` / `c7_*` per lane.
- HLLD `_solve_and_ray` resolves those answers' slow fans in the total pressure with their own shock masks. `LAST_DIAG` gains `coplanar7_mask` / `_stage` / `_rot`.
- `replay_pop.py` records src 4.
- Tests: `test_coplanar7.py` (5) pass. Lanes 1013 / 214 from an unrotated seed reach the regular answer in stage 2. With the switch off there are no new keys.

**Batched starts (`RMHD_PLANAR4_STAR_BATCH=1`, HLLD `_planar4_star_batched`, and the rotation choices in `_planar4_rotation`).** Every start of every lane runs in one solve per difference scheme; per lane the first accepted start in the serial order is kept. The solves are batch-independent, so the answers are the serial loop's. Local smoke test (24 ledger faces, fixes + gate + rotation, no pool): exact flags, rungs and fluxes are **bitwise equal**, wall 93 s vs 132 s (-30%).

## The course updated with this week's findings (2026-11-05)

User: "yes, update the course with this week's findings". `HLLD/docs/course/rmhd_riemann_course.tex` and the PDF, rebuilt (69 pages, no unresolved references, no overfull lines; auxiliary files removed). Uncommitted. The PDF before the change is `course_before_2026-11-05.pdf`.

What changed:
- **4.5 Coplanar data and the Alfven codebox; 7.2.3; Fig. 7.1 caption.** The non-trivial coplanar turn reverses b^y but may keep the laboratory sign of B^y (B^y = W(b^y - b^0 v^y)). Face 1013 is the example. Production's psi + pi asks only for the reversal; `coplanar_rotation` takes both.
- **4.7 keybox "are the exact answers solutions, and are they admissible?"** The verifier, the 256k table and the gate.
- **7.4.** planar4's kind S, the extra starts and the turn branch. A new paragraph on the planar seven-wave stage, with the 79610 replay. The pipeline gains the November items.
- **8.2 keybox "two exact answers, one flux"** with Fig. 8.1 (pgfplots: face 1013, refused vs regular) and Exercise 8.4 (third-order agreement of jump and fan).
- **8.4.** The MARGIN bullet now carries the measured 2.0% / 1,350 entropy-falling jumps.
- **8.5.** The four-quadrant ledger: types A, B weak and B strong were solver defects; 860 / 872 admissible; 12 left.

**The planar stage AFTER the six-unknown Newton (`RMHD_COPLANAR7=after`).** In the 64^2 short runs the planar stage FIRST cut the wall per step against the rotation runs on Orszag-Tang (4.3x -> 1.6x the control, step 28) and on the rotor (2.9x -> 1.8x, step 8). It slowed the four-quadrant problem's easy early steps from 1.0x to 4.6x (step 16), because it replaces a Newton that converges in a few cheap iterations. Run 79611 was cancelled after 17 steps.

The after mode runs the planar stage only on the coplanar faces the Newton loses, or answers with a jump across which the entropy falls (`coplanar7_b.entropy_min7`; `api.solve_batch` refactored into `take7`). Tests: `test_coplanar7.py` 6/6. Calea worktree rmhd_final-loose8 + HLLD-fix9; launcher `calea_c7arun64.sh`.

Short tests, all writing `*_64_exact_<mode>_shortgate`:

| problem | after (finalrotc7a) | after + batched starts (finalrotc7ab) | steps |
|---|---|---|---|
| 4Q | 79643 | 79646 | 150 |
| OT | 79644 | 79647 | 80 |
| rotor | 79645 | 79648 | 80 |

### Why the four-quadrant run was 4x slower, and the late rung (2026-11-05)

Wall at the same step against the control (`same_step.py`, 19:17-19:40): OT after + batched 1.26-1.58x (rotation run 4.3-4.7x), rotor 2.0x (2.9x), but the 4Q 3.7x (rotation run 1.0x). Per step (`diag_4q/step_incr.py`) every planar-stage mode cost a flat 60-80 s from step 2 on, against 8-40 s for the control, while the sweeps held only 135-208 exact faces; first and after cost the same.

The coverage maps (`diag_4q/cov_steps.py`) say why: in the control and the rotation run about 190 faces a step (6 sweeps) are answered by the five-wave planar solver; in every planar-stage run the planar seven-wave stage answers those same faces instead. They are faces nearly at rest (median W 1.00; 1 of 96 above W = 2; corrected 11-05 21:00, first written as 'W ~ 7 flows') in the 45-degree field (B_n = B_t = 0.35), with the seven wave speeds about 0.25 c apart, where production's network (`ml_guess_rotor_ft.pt`) gives a seed the six-unknown Newton cannot start from (0 iterations). The 4Q network it3 starts the Newton there fine (3-4 iterations). Replayed locally (`diag_4q/run_p5.sh`, 96 such faces from the control's harvest, production network):

| arm | answered by | wall |
|---|---|---|
| stage off | five-wave planar, 4 iterations | 19 s |
| after | planar seven-wave, 94 at stage 3 (pressure-steered seed) after ~30 iterations; no rotation | 79 s |

Flux after vs off: median 3.3e-10, max 2.9e-9. The stage buys nothing on these faces. (A first local replay with the 4Q network it3 reproduced nothing: with it3 the Newton converges on them.)

**Late rung** (`RMHD_COPLANAR7=late`): the planar seven-wave stage leaves `api.solve_batch` (`coplanar7_b.late()`, `wanted()` excludes it) and becomes a rescue rung of the exact flux, `exact_flux._coplanar7_and_ray`, between the five-wave planar solver and the four-unknown one, on the worker pool. It gets the COPLANAR lanes neither the Newton nor the five-wave solver answered admissibly, from the seeds the sweep's Newton started from, and holds its answers to the same acceptance: physical state, wave order, the gate. The Newton's entropy-lowering answers reach it through the gate. Coverage flag `coplanar7` (harvest.py), diagnostics `coplanar7_late_*` in LAST_DIAG, `replay_pop.py` labels its answers src 4. Not harvested as rows, as for the four-unknown answers.

Local: the 96 faces 19 s (= off); the 24-face ledger smoke: the same 8 refused faces answered at the same stages as the first mode, flux identical (max 1.5e-8), 13/13 controls. Tests `test_coplanar7.py` 7/7.

Calea: worktrees rmhd_final-loose9 + HLLD-fix10 (`c7_patches/setup_late.sh`, `c7l_*.diff`, `coplanar7_b_late.py`); paired ledger replay 79680 (`calea_replay_c7l.sh`: finalrot / first / after / late, batched starts in every arm); short runs late + batched starts 79681 (4Q, 150 steps), 79682 (OT, 80), 79683 (rotor, 80), launcher `calea_c7lrun64.sh`, tag `finalrotc7lb_shortgate`. Cancelled as superseded: 79643-79646 (after mode, both 4Q runs and the OT/rotor runs without the batched starts); kept 79647/79648 (OT/rotor after + batched) as the comparison.

**Results (20:35).** Paired ledger replay 79680 (2,100 faces, batched starts in every arm): rotation only 860/872 + 1228/1228 in 855 s; first 861/872 + 1228/1228 in 554 s; after 861 + 1228 in 746 s; late 861 + 1228 in 792 s. On these hard faces first is fastest, late 7% faster than rotation only.

Short runs, cumulative wall against the control at the same step (`same_step.py`) and faces lost over the steps every run has done (`diag_4q/cov_sum.py`):

| | 4Q (step 98 / lost over 1-20) | OT (step 59 / lost over 1-50) | rotor (step 46 / lost over 1-40) |
|---|---|---|---|
| control | 1.00x / 64 (0.31%) | 1.00x / 364 (1.52%) | 1.00x / 583 (1.61%) |
| rotation run | 0.92x / 64 | 4.10x / 135 (0.56%) | 2.60x / 128 (0.35%) |
| planar first | 3.13x (at 30) / 64 | 1.66x / 26 (0.11%) | 1.47x / 48 (0.13%) |
| after + batched | 3.29x (at 29) / 64 | 2.67x / 83 (0.35%) | 2.17x / 54 (0.15%) |
| **late + batched** | **0.89x / 64** | **1.70x / 26 (0.11%)** | **1.28x / 46 (0.13%)** |

The late rung answers 794 OT and 2,502 rotor faces (the four-unknown solver is left 63 and 105, against 748 and 2,523 in the rotation run); in the 4Q's first 20 steps it never fires. Late + batched is the cheapest or within 3% of it on every problem and loses the fewest faces. Cancelled 79647/79648 (after + batched, superseded) and, on the user's word (11-05: "yes, cancel 79441 and 79442"), the OT/rotor rotation runs 79441/79442; full runs launched: 81297 (OT), 81298 (rotor), 81299 (4Q), tag `finalrotc7lbgate`; the 4Q rotation run 79440 finishes.

**2026-11-06: the 4Q past step 100, and what the gain is made of.** The 4Q rotation run 79440 finished (369 steps): lost 436 of 1,078,032 (99.960%), 14.25 h, L1(rho) vs 512^2 0.1022 -- against the loss fixes alone (78636) 402 (99.963%), 11.50 h, 0.1021, and production 7,970 (99.261%), 8.26 h, 0.1023 (`compare_late.sh r2d` -> `cmp_late_r2d/`). The gate moves 6,628 Newton answers to the four-unknown solver (16,204 against 8,074) and costs 34 faces.

The claim above ("cheapest or within 3% on every problem") held for the 4Q only up to step 100. Wall per 25 steps against the control (`diag_4q/step_bins.py`):

| steps | fixes | fixes + gate | rotation | late + batched |
|---|---|---|---|---|
| 1-100 | 0.82-0.98x | 0.84-1.01x | 0.83-1.00x | 0.82-1.01x |
| 101-125 | 1.11x | 1.09x | 1.09x | 1.45x |
| 126-150 | 1.40x | 1.48x | 1.40x | 1.92x |
| 151-200 | 1.17-1.28x | 1.46-1.53x | 1.42-1.51x | 1.87-1.88x |
| 201-250 | 1.55-1.60x | 1.71-1.96x | 1.64-1.83x | 2.21x |
| 251-275 | 2.10x | - | 2.84x | 2.73x |

At step 275 the late run has used 34.8k s against the rotation run's 29.6k (1.18x); over steps 271-276 it is the cheaper one (275 against 337 s a step). Its rung answers about three quarters of what the four-unknown solver answered in the rotation run (226-250: 1,230 + 412 against 1,632; `diag_4q/cov_bins.py`), so in most sweeps both rungs run on a handful of lanes, and a sweep's rescue costs its slowest lane's iterations, not its lane count. Lost over steps 1-275: 324 against 360.

Confound: the late runs have the batched starts (`RMHD_PLANAR4_STAR_BATCH`), the rotation runs do not, so part of the OT/rotor gain may be the batching. Timing arms (`cost_arms/`: ten steps from one restart state per problem, the run's settings, wall per stage from `tpatch.py`, which wraps `exact_pool.map_lanes` and `exact_flux._run_pipeline`): 81475-81478 (4Q from the late run's step 270: rot, rotb, lateb, latebnp), 81481 (4Q from the short run's step 150, the three arms in turn), 81479 (OT from step 340), 81480 (rotor from step 180, six steps).

**Timing arms, first results (11-06).** Ten steps from one restart state per arm, the run's settings, wall per stage (`cost_arms/cost_report.py`, `stats_report.py`).

4Q from the late run's step 270 (81475-81478), steps 271-280:

| arm | wall 272-280 | four-unknown | planar seven-wave | lost |
|---|---|---|---|---|
| rot (79440's switches) | 3,032 s | 2,490 s / 940 faces | - | 46 |
| rotb (+ batched starts) | **2,026 s** | 1,311 s / 940 | - | 46 |
| lateb (81299's switches) | 2,598 s | 1,192 s / 138 | 792 s / 904 offered, 798 answered | 44 |
| latebnp (lateb without the pressure stage) | 2,483 s | 1,271 s / 436 | 571 s / 898, 496 | 44 |

The batched starts alone take a third off the step (same answers). The late rung answers 88% of what it is offered, yet the four-unknown rung's wall hardly moves (1,311 -> 1,192 s with 7x fewer faces): it still runs in every sweep, ~20 s a sweep, so the late rung's 792 s is added on top (+28% against rotb, for 2 faces fewer lost).

4Q late rung from the short run's step 150 with per-lane statistics (81482, `tpatch2.py`), steps 151-160: Newton 416 s, five-wave + ladder 104 s, planar seven-wave 505 s (60 calls, 90 faces), four-unknown 271 s (40 faces) of 1,386 s. The rung answered 50: 12 at stage 1 (the network seed, ~3 iterations), 38 at stage 3 (pressure-steered) after stages 1-2 failed; 34 of the 40 it did not answer the four-unknown solver did, 6 were lost. Every in-process planar4 solve costs ~2.2 s whatever its lanes (c7: 220 solves, 956 branch-lanes, 492 s).

**One-call rung (user 11-06: "yes, start the dataset job and the one-call version").** `RMHD_COPLANAR7_CALLS` = 4 (staged, default) / 2 (stage 1, then 2-4 together) / 1 (every branch of every stage in one solve), `rmhd/batched/coplanar7_b.py`: each lane's branches stay in preference order and the first accepted wins, so the answers are the staged ones; `stage` and `n_iter` are counted as staged. Tests: `test_merged_calls_give_the_staged_answers[2,1]` (bitwise on lanes 1013/214, both stage 2), 9/9 pass; the 24-face ledger smoke through HLLD: fluxes, rungs and stages bitwise equal to staged and to 11-05's run (`planar_ml/run_onecall_smoke.sh`; answers at stages 1/2/3). Timing arms 81483 (4Q step 270) and 81484 (step 150): lateb1, lateb2 (`cost_arms/calea_cost_arms4.sh`).

**Planar dataset.** `RMHD_HARVEST_RESCUES=1` (HLLD `exact_flux` + `harvest`, off by default): the late rung's answers are harvested as source=2 and the four-unknown solver's as source=3, and every solved row gets `steer` (k,2: each side's slow-wave steering, planar4_b KIND_*), `rot` (k,2) and `stage` (k); off, the schema is unchanged. Local smoke (`planar_ml/run_harvest_smoke.sh`): 4 source-0, 11 source-2 (stages 1-3, steerings signed and pressure, rotations) and 6 source-3 rows for the 21 exact faces. Populations (`planar_ml/pop_refused.py`: the harvest's unsolved rows + its source-0 answers whose entropy falls by more than 1e-9): 4Q 64^2 ctl1102 141,872 (137,606 + 4,266), 4Q 128^2 it3 480,918 (451,342 + 29,576), OT 64^2 ssretry 146,596 (120,268 + 26,328), rotor 64^2 ssretry 80,462 (63,907 + 16,555). The gated 4Q rotation run's harvest misses the ~20k answers its gate refused (8 entropy-falling source-0 rows; attempted - seven = 163,222 against 143,502 unsolved rows), hence ctl1102. Jobs 81485-81488 (`planar_ml/calea_planar_dataset.sh` -> `build_dataset.py`: chunks of 4,096 consecutive faces, the full chain with CALLS=1, a manifest per chunk; resumable) -> `results/planar_ml/<run>/`. Calea worktrees rmhd_final-loose10 + HLLD-fix11 (`c7_patches/setup_onecall.sh`, `hr_hlld.diff`, `coplanar7_b_onecall.py`).

**Correction (11-06): the gated harvest is complete; the reclassified lanes are not.** The rotation run's unsolved rows DO hold its gate refusals (reason 3: 14,026 against the control's 7,732, the +6.3k the gate moved), so the note above about the gated harvest is wrong. The rows missing from EVERY run's harvest (4Q 64^2: attempted - seven - degenerate = 162,354 against 143,502 unsolved rows, ~18.9k; the control is the same) are the lanes whose seven-wave attempt solve_batch reclassified as degenerate and whose three-wave answer exact_flux refuses AFTER the harvest (`reclass`): counted answered when `record` ran, they are written nowhere, then go on to the rescues. HLLD fix (`hr2_hlld.diff`, under RMHD_HARVEST_RESCUES): `Harvester.record_unsolved` writes them as unsolved rows with reason 4 (R_RECLASSIFIED); tests `tests/test_harvest_rescues.py` (3) + the harvest tests with the switch off and on (7 + 7). The running dataset jobs (fix11) do not have it: their populations come from the old harvests and miss these lanes (~12% of what the planar rungs see).

**Timing arms, 4Q from step 150 (81481), steps 151-160:** rot 1,063 s (four-unknown 587 s / 90 faces), rotb 1,009 s (538 s), lateb 1,218 s (four-unknown 270 s / 40, late rung 515 s / 90 offered, 50 answered); lost 6 in every arm; the stats arm repeats lateb to 1,210 s with the same faces. At ~1.5 faces a sweep the batched starts give 5%, the late rung costs +21%.

**Harvest runs (user 11-06: "yes, launch the harvest runs when the late runs finish").** `calea_harvest64.sh` MODE=finalrotc7l1h: the late runs' switches + RMHD_COPLANAR7_CALLS=1 + RMHD_HARVEST_RESCUES=1, worktrees rmhd_final-loose10 + HLLD-fix12 (`c7_patches/setup_harvest.sh`: 51e9c67 + hr2_hlld.diff), pilot with the switches off against the control, tag `finalrotc7l1hgate`. Submitted with Slurm dependencies: 81489 4Q after 81299, 81490 OT after 81297, 81491 rotor after 81298 (afterany). Their diag.csv should match the late runs' row for row (neither switch changes an answer) -- a full-run check of the one-call rung on calea.

**Planar network training (user 11-06: "start the planar model training when the data is in").** `planar_ml/planar_data.py` (rows of the dataset jobs -> the planar canonical frame: rotate so the left tangential field is +y, scale by rho_L; 13 features; targets ln p_LF, ln p_RF, By_CD, ln P_CD, the rotation per side (the late rung's flag, else the Alfven zone differs from the fast one), the pressure steering per side (sources 2, 3 only); every row also mirrored x -> -x; time per row from the source run's coverage sweeps and diag.csv) and `planar_ml/planar_train.py` (MLP 13-256-256-128-8, Huber + BCE with balanced positives, weights late rung 5 / four-unknown 20, split by t/t_end: train < 0.7, validation to 0.8, test after, rows and mirrors together; early stopping). Pilot on the first chunks (139k rows, 8 epochs, local): the pipeline runs; numbers not meaningful. Job 81494 (`calea_planar_train.sh`, afterok on 81485-81488): data_v1 + two trainings (classification weight 1 / 0.3) -> `results/planar_ml/train_v1/`. In the one-call rung the classifier matters less than the seed: the model's seed can be offered in every branch at little cost; the classifier orders them for the staged rung and picks the four-unknown solver's start.
Gamma is now the planar network's 14th input (11-06, before job 81494 starts; the scripts are read at run time): the dataset mixes 5/3 (4Q, rotor) and 4/3 (OT), and the planar answer depends on it. The seven-wave networks leave gamma out and are one per EOS ([[rotor-ot-proper-retrain]]); the planar network is meant to serve all three problems. Pilot re-run on the rotor + OT chunks so far: 218,438 rows, X (n, 14), gamma 5/3 and 4/3 by run.

**One-call rung: NEGATIVE (timing arms 81483/81484, 11-06).** Same faces and diag.csv bitwise equal to the staged rung (timing columns excluded) at both windows -- the merge is exact on calea too -- but the rung costs MORE: 4Q steps 271-280 staged 792 s, two calls 936 s, one call 950 s; steps 151-160 515 / 584 / 695 s. The per-call floor is not the whole cost: a merged solve runs every branch, the failing ones until they stall, and each iteration's work grows with the rows (up to 16 branches a lane). RMHD_COPLANAR7_CALLS stays 4 (staged). The harvest runs 81489-81491 (CALLS=1) were cancelled before they started and resubmitted with the staged rung: MODE=finalrotc7lh, tag `finalrotc7lhgate`, 81497 4Q after 81299, 81498 OT after 81297, 81499 rotor after 81298. The dataset jobs keep CALLS=1 (same answers, a little slower).

Timing arms OT (81479, steps 341-350) and rotor (81480, 181-186) against rotb: OT late + batched 1,238 s vs 1,699 s (-27%), lost 7 vs 69 (the four-unknown solver runs in 9 of 60 sweeps); rotor 2,606 vs 2,119 s (+23%), lost 77 vs 117.

**11-06 afternoon.** 4Q late run 81299 DONE (11:39, 14.95 h): lost 396 of 1,078,224 (99.963%), L1(rho) 0.1021 -- rotation run 436 / 14.25 h / 0.1022, fixes alone 402 / 11.50 h / 0.1021, production 7,970 / 8.26 h / 0.1023; four-unknown answers 2,242 (rotation run 16,204), the late rung ~14k (`compare_late.sh r2d late` -> `cmp_late_r2d/`). Harvest run hv-r2d 81497 (started 11:40): pilot bitwise against the control, and diag.csv rows 1-213 identical to the late run's (timing columns excluded), same pace (18,790 s at step 212 vs 18,825 s). Dataset jobs all done: 4Q 64^2 1.7 h, OT 3.5 h, rotor 3.7 h, 4Q 128^2 7.7 h. Planar training v1 (81494, 15:26-15:27): data_v1 1,689,402 rows with mirrors (base: seven-wave 7,669, five-wave 623,019, late rung 186,026, four-unknown 27,987); both arms EARLY-STOPPED at epoch 32 (~40 s) because the validation rotation loss rose from the start while the regression kept improving. Test block (t/t_end >= 0.8), late-rung faces: pressure steering 96.8% right (always-the-commonest 78.9%; recall 96%, false alarms 2.6%); rotation 69% (commonest 78%; balanced weights over-predict, recall 88%, false 19%); ln p_LF / ln p_RF / ln P_CD mean errors 0.06-0.07, By_CD 0.065-0.093 against a typical |By_CD| of 0.31. Not yet compared with the current seed (rotor_ft converted) on the same faces.

**11-07 08:53 (user: "check calea").** All three late runs DONE: OT 81297 (23:04, 26.4 h) lost 988 of 4,413,369 (99.978%), L1(rho) vs 512^2 0.2647 -- production 33,153 (99.240%), 15.16 h, 0.2657; rotor 81298 (22:41, 26.0 h) lost 3,591 of 963,257 (99.627%), L1 0.2874 -- production 12,622 (98.690%), 13.81 h, 0.2874; 4Q (above) 396 (99.963%), 14.95 h, 0.1021 -- production 7,970, 8.26 h, 0.1023. Lost maps / error maps / walls: `cmp_late_{r2d,ot,rotor}/` (production vs new; the 4-run versions were unreadable, titles overlapped). Rotor: the new chain's losses sit in the last quarter (2,265 of 3,591) in two bands across the core; production's same window lost 5,632. OT: 0 / 177 / 246 / 323 per quarter (physical faces) against 346 / 7,246 / 6,567 / 11,623. Harvest runs: hv-r2d 81497 DONE (02:38, 14.98 h, diag 369/369 identical to the late run); hv-ot 81498 (from 23:04) step 337/655 and hv-rotor 81499 (from 22:41) 189/284, both diag-identical so far, pilots bitwise; expected ~Sun 01:00-01:30. 4Q 512^2 77227: step 1293, t 0.174 of 0.4, ~360 s a step and rising -> ~Nov 14-16. Not ours: 84379 rns-grid-test (calea01, 08:47). Disk 82% (7.7T free), 9 nodes idle.

## Cost levers (user 11-07: "Do these two levers. Be ambitious. But watch out that the plan does not change")

Both bit for bit, both off by default (the running harvest runs, the 512^2 run and the planar network plan are untouched).

**Lever 1 -- the planar solves' numpy root-finders compiled (`RMHD_PLANAR_NJIT`=0/1/check, rmhd `rmhd/batched/planar_njit.py`).** Profile of `exact_flux_batched` on 3-6 hard faces of the 64^2 runs (`levers/profile_rung.py`): the ~2 s per planar solve call is not Python glue but three numpy root-finders that evaluate a residual dozens of times on a few lanes -- the pressure-steered slow shock (`planar4_b.slow_shock_p`: 501-node scan + bisections; 2.3 of 6.0 s on 3 4Q faces, 6.0 of 14.0 s on 3 rotor faces), the fast shock's edge scan (`shock_b.fast_edge_scan`: 3.1 of the rotor's 14.0 s) and the slow shock's retry (`slow_shock_b.retry`: 3.0 of 13.3 s on 6 4Q faces). Each is transcribed per lane into one prange kernel on shock_njit's / slow_shock_njit's compiled residual, post-shock state, bisection and attempt; numpy's once-per-call parts (windows, node fractions) stay numpy and are passed in. `check` runs both and counts differing lanes (`planar_njit.CHECK`, dump dir `RMHD_PLANAR_NJIT_DUMP`). Local: 0 of 14,434 lanes differ (slow_shock_p 7,667, retry 6,485, edge scan 282); wall on the profile faces 6.0 -> 2.2 s (4Q), 14.0 -> 4.3 s (rotor), 13.3 -> 6.1 s (4Q, 6 faces); OT unchanged (its time is the Newton and the five-wave ladder). TRIED AND DROPPED: `fullcontact_b.make_xi` through fan_njit's compiled xi -- its quartic differs from numpy's by a few ulp (median 6e-16, max 1e-14 relative, never in which roots exist), so it would move every answer at round-off. Left: the compiled fan integrator (fixed-step RK4, real work). Tests: `tests/batched/test_planar_njit.py` (3) + the planar/slow-shock/shock suites in all modes (only the two known failures: the shock timing test and test_planar4 constructed answer). Calea worktree rmhd_final-loose11 (`c7_patches/setup_lever.sh`, `lev_rmhd.diff`, `lev_files/`); timing arms 84463-84466 (`cost_arms/calea_cost_arms5.sh`: lateb / latebnj / latebck from the step-270 and step-150 4Q, OT 340 and rotor 180 restarts; the check arms write njit_check.json).

**Lever 2 -- one seven-wave pipeline call per RK stage (`RMHD_MERGE_SWEEPS`, HLLD `src/physics/sweep_merge.py`, `driver2d.ct_stage`, `exact_flux_batched(_rendezvous=)`).** The pipeline (Newton + retries + ray) costs a sweep about the same whatever its lane count (~7 s for ~470 lanes on the 4Q, ~9.3 s for ~1,660 on OT). Merged: each sweep runs in its own thread, one at a time -- x to its pipeline call, then y to its own, the merged pipeline once on all lanes (concatenated per lane, split back; `n_fan` recounted per sweep), then x to the end, `done`, then y. Everything that works on a whole sweep stays per sweep and in order (reconstruction, HLLD fallback and star pressure in torch, gates, network seeds, rescue rungs, flux assembly, harvest records and coverage); `LAST_DIAG` is saved at the hand-over and restored on resume. Local, short 2D runs merged vs unmerged (`levers/run_merge_test.sh`: 4Q 32^2 3 steps production switches; rotor and OT 32^2 4 steps late chain + compiled kernels + rescue harvest; rotor through a 4-worker pool): restart state, diag.csv and every harvest file identical. Test `tests/test_sweep_merge.py` (2); 59/59 of the flux/CT/pool/harvest suites pass with the merge off and on. Calea worktree HLLD-fix13 (`c7_patches/setup_lever2.sh`, `lev2_hlld.diff`, `lev2_files/`); timing arms 84480-84483 (`cost_arms/calea_cost_arms6.sh`: latebnj vs latebnjm from the same restarts).

**Cost levers -- calea results (84463-84466, 84480-84483; 11-07 14:56).** Wall of the steps after the first (start-up excluded), same restart state per row; production from its full run over the same steps (another node, indicative):

| window | production | new chain, numpy | + lever 1 | + lever 1 + 2 |
|---|---|---|---|---|
| 4Q 272-280 | 918 s | 2,655 s | 851 s | 793 s |
| 4Q 152-160 | 713 s | 1,216 s | 525 s | 483 s |
| OT 342-350 | 1,022 s | 1,239 s | 733 s | 689 s |
| rotor 182-186 | 1,490 s | 2,605 s | 597 s | 553 s |

Lever 1 also speeds the seven-wave Newton (its residual uses the slow-shock retry and the fast edge scan): 4Q 270 551 -> 362 s, rotor 727 -> 224 s; lever 2 then takes the Newton stage 362 -> 272 s (4Q 270), 233 -> 168 (4Q 150), 397 -> 328 (OT), 224 -> 175 (rotor) with half the calls.

Bitwise on calea: **lever 2 yes** (diag.csv identical, every harvest file byte-identical, all four windows). **Lever 1 NO** (it was on the Mac): check runs 4Q 270 slow_shock_p 6,394 / 41,998 lanes differ, fast_edge_scan 1,158 / 3,176, slow_retry 0 / 272; rotor 108 / 397, 39 / 241, 0 / 81. A few faces change rung (4Q 270 four-unknown 94 -> 96; lost 44 -> 44, OT 7 -> 8, rotor 77 -> 76); after 10 steps ~65% of cells differ by up to ~1e-6 relative. Cause measured (calea 84496, `levers/mathcheck.py`): numpy 2.5.3 on the AVX-512 nodes and numba disagree by 1 ulp on pow (5.3% of values), exp (4.6%), log (0.3%); sqrt agrees. On the Mac both use the same libm. The production kernels (RMHD_SHOCK/SLOWSHOCK/FAN=njit) already differ from their numpy twins on calea in the same way; lever 1 extends that to three more routines. Decision (user): take lever 1 with round-off-level differences, or keep strict bitwise identity with the existing runs (lever 2 only).

**Decision (user 11-07: "take those new kernels . If they are faster and smarter, very good").** Both levers are in the production kernel set of the patch set: `scripts/kernels.env` exports `RMHD_PLANAR_NJIT=1` and `RMHD_MERGE_SWEEPS=1` (overridable), with the calea round-off caveat written there (and the old "bitwise against its numpy twin" lines qualified: true on the Mac only). Applied to rmhd_final-loose11 on calea (`lev_rmhd.diff` regenerated); every launcher that sources `$RMHD/scripts/kernels.env` with PYTHONPATH=loose11 and HLLD-fix13 now runs both. The code defaults stay numpy / unmerged (the reference). Not changed: the running harvest runs hv-ot 81498 / hv-rotor 81499 (loose10 / fix12) and the 4Q 512^2 run 77227. Uncommitted.

**4Q 512^2 resumed with the new chain (user 11-07: "yes cancel, and rerun, do we have checkpoints?").** 77227 (r2d-ft3-512: ft3, production's chain of 11-03, every face harvested, HLLD-harvest worktree) writes a full restart every 10 steps (the latest only) plus 40 snapshots. `cancel_77227_at_restart.sh` cancels it as soon as its restart reaches step 1380 (t ~0.186, 46%), so no step is lost. 84516 (`calea_r2d512_new.sh`, afterany:77227): (1) a 2-step pilot from a saved copy of the step-1370 restart (`results/r2d_512_exact_ml_r2d_it3_newchain/pilot_restart_1370.npz`) with every new switch off, its diag rows against the original run's 1371-1372 bitwise; (2) the run from the step-1380 restart with the new chain -- loss fixes, gate, rotation re-solve, late rung, batched starts, the complete harvest (RMHD_HARVEST_RESCUES) and the kernel set of rmhd_final-loose11 (RMHD_PLANAR_NJIT, RMHD_MERGE_SWEEPS); ft3 + s42/s47 as before; NSNAP 40, harvest cap 2e9, 200k-row shards. Worktree HLLD-fix14 (`c7_patches/setup_512.sh` = fix13 + patch_harvest_max.py). Output `results/r2d_512_exact_ml_r2d_it3_newchain` (the first 1380 steps stay in the original directory).

**512^2 resume, first numbers (11-07 18:35).** Pilot bitwise; the run left step 1380 at 17:59. Steps 1381-1387: 265-273 s each (the old chain's last 180 steps averaged 373 s; its last step 337 s). First harvest shard (sweeps 8280-8318, steps 1381-1387): 205,123 attempted, 205,050 exact -- 73 lost (0.036%) -- against the old chain's last six shards (sweeps 8100-8279, steps 1351-1380): 916,180 attempted, 1,912 lost (0.209%), so ~6x fewer lost faces with the gate's refusals counted as lost. Answered by: seven-wave 186,577, five-wave 15,297, late rung 1,412, four-unknown 887. The harvest has every row type: solved rows of sources 0-3 with steer/rot/stage, unsolved reasons [16,070, 0, 0, 1,131, 468] (468 reclassified lanes, reason 4). At ~270 s a step the remaining ~1,580 steps (t 0.187 -> 0.4) take ~5 days; the old run's step cost grew ~1.5x from step 300 to 1380, so plan on ~7.

**Planar network v2 (queued 11-07 18:50, job 84559, afterok:81498:81499).** `planar_ml/calea_planar_train2.sh` -> `results/planar_ml/train_v2/`:
- data: v1's four dataset jobs plus the three harvest runs, read as `harvest:RUN` (`planar_data.load_harvest`: sources 1-3 only -- the Newton's own answers are faces the planar rungs never see -- one chunk per shard, time from the shard's coverage). Every row now keeps its solver-frame states (`UL`, `UR`).
- baseline: `planar_ml/planar_baseline.py` computes production's seed (rotor_ft candidate 0, clamped, exact_flux._ml_seeds -> coplanar7_b.planar_seed; ln P_CD from planar4's structure at that seed, as the pressure-steered stage builds it) on the validation and test rows, in the targets' units, through planar_data.rows_to_arrays. The rebuilt features equal the stored ones (max |dX| = 0, local), and 32 spawned workers give the serial answer bit for bit. Cost ~3 ms a row on one core, nearly all planar4's fast-fan integration.
- training: v2a 256-256-128 and v2b 512-512-256 stop on the REGRESSION validation loss (`--stop-on reg`), classification weight 0.3, 300 epochs, patience 40. The report gives each target's median and 90th-percentile error for the network and the baseline on the same rows, how often each is closer on all three signed-field unknowns, and the baseline's rotation accuracy.
- local test on three 4Q harvest shards (8,914 faces; `claude_scratch/planar_v2_test`): pipeline end to end. On those late-rung faces production's seed is already close: median error ln p_LF 0.029, ln p_RF 0.047, By_CD 0.096, ln P_CD 0.025 (90th percentiles 0.10 / 0.11 / 0.26 / 0.17). The v1 network's MEAN errors on its test block were 0.06-0.07 in the pressures, so the regression may not beat the current seed by much; the steering head (which stage answers) looks like the part with something to give. Decide on train_v2's paired numbers.

**Planar network v2 RESULT (84559, 11-08 01:24-01:47).** data_v2: 2,637,294 rows with mirrors (base: seven-wave 7,669, five-wave 947,835, late rung 316,241, four-unknown 46,902). Baseline on the 1,175,604 validation + test rows: 599 s on 32 workers, rebuilt features equal (max |dX| 0). v2a (256-256-128) ran all 300 epochs (best regression epoch 273, val 6.3e-4); v2b (512-512-256) stopped at 94 (best 54, val 8.5e-4) and is worse everywhere -> **v2a**. Test block (t/t_end >= 0.8), late-rung faces (223,394 rows), median |error| network vs production's seed (90th percentiles): ln p_LF 0.023 vs 0.047 (0.083 vs 0.105), ln p_RF 0.023 vs 0.046 (0.082 vs 0.103), By_CD 0.022 vs 0.024 (0.094 vs 0.177), ln P_CD 0.022 vs 0.046 (0.077 vs 0.115); closer on all three signed-field unknowns 31% vs 13%. Steering 95.4% right (always-field 80.8%; recall 91%, false 3%), rotations 84.2% (production's suggestion 71.3%, majority 76.6%). Four-unknown faces (39,460): pressures 3-5x closer (0.018-0.020 vs 0.064-0.102), By_CD equal (0.012 vs 0.011, p90 0.045 vs 0.21); its steering there is WORSE than always-pressure (82.7% vs 91.3%) -- not to be used for the four-unknown solver.

**v2a in the late rung (11-08, uncommitted, off by default).** rmhd `rmhd/batched/planar_net.py` (inference: planar_data's features operation for operation, outputs mapped into to_planar's frame) + `coplanar7_b.py`: `RMHD_COPLANAR7_NET=<checkpoint>` builds the stages around the network -- 1 its seed in its branch (steering + rotations), 2 its seed with the other rotations, 3 its seed with the other steering, 4 what production would have tried (`RMHD_COPLANAR7_NET_FALLBACK=all`: every production branch; `first`: production's first branch + the extra checkpoints). Non-finite network output -> production's seed in its place. `coplanar7_b.STATS` counts calls, lanes, planar4 solves and (lane, branch) pairs (for replays). Tests `tests/batched/test_planar_net.py` (off by default; frame-free features; unusable output = production bit for bit; the ledger lanes still get their regular answers with the network). Calea: worktree rmhd_final-loose12 (`c7_patches/setup_net.sh`: loose11's tree + `net_files/` + data/ml_planar_v2a.pt; loose11 untouched for the 512^2 run).

**Paired replay (`planar_ml/replay_net.py`, `calea_replay_net.sh`, `compare_replay.py`).** Population: the three harvest runs' test block, every sweep's late-rung faces (source 2 + source 3 rows, split into sweeps by the coverage counts -- they match in all 69 + 242 + 61 shards): 4Q 8,850 faces, OT 30,185, rotor 10,774 (lost faces have no row and are not in it). Each sweep through exact_flux_batched with the 512^2 run's chain + kernels.env, production seeds (rotor_ft + s42/s47), no pool; arms off / net / netfirst, one node each.
Local tests (11-08): test_planar_net.py 6/6 and test_coplanar7.py 9/9. The unusable-output test caught a closure bug in the first version: the network branch's seed-index array was named `first`, which the stage loop reassigns (np.unique return_index) after stage 1. Stages 2-4 would have read garbage seed indices; it is now `s_first` / `s_other`. Ledger lanes 1013 / 214: the network answers both at stage 1 in 4 iterations each; production answers at stage 2. Pilot 84574 (net, 6 4Q test sweeps, 94 faces): all answered at stage 1, one planar4 solve per sweep, late rung 0.7-0.9 s per sweep; a sweep costs ~18 s in all, the Newton (7-9 s) and the five-wave solver + ladder (9-12 s) being the same in every arm. Full arms 84575 off / 84576 net / 84577 netfirst (11-08 02:50), each with 4 parallel parts (4Q, OT halves, rotor; 16 numba threads each) merged into `results/planar_ml/replay_v2/<arm>.npz`; `compare_replay.py off.npz net.npz netfirst.npz`.

**Replay RESULT (84575-84577, 11-08 02:50-07:06, 4h20 each).** 50,104 held-out faces in 1,590 sweeps (`compare_replay.py`):

| arm | late rung | at stage 1 | four-unknown | lost | solves / sweep | pairs / sweep | late + four wall | whole replay |
|---|---|---|---|---|---|---|---|---|
| off (production) | 41,875 | 21,735 | 8,130 | 34 | 3.53 | 163.5 | 15,582 s | 59,071 s |
| net, fallback all | 46,027 | 37,500 | 3,982 | 30 | 3.13 | 132.1 | 12,964 s | 55,905 s |
| net, fallback first | 44,148 | 37,500 | 5,439 | 452 | 3.13 | 106.7 | 12,884 s | 55,956 s |

- Off reproduces the harvest's rung on 99.47% of faces.
- Fallback `first` turns 422 of production's late-rung answers into lost faces: REJECTED.
- Fallback `all` changes a rung only as four-unknown -> late rung (4,148) or lost -> late rung (4, rotor).
- By problem: 4Q late-rung wall 3,302 -> 1,110 s and solves/sweep 3.84 -> 2.28. OT late rung about the same (+6%), four-unknown load halved. Rotor: every sweep still runs all 4 stages (always an unanswerable lane); four-unknown 5,343 -> 2,643 faces.
- The Newton + five-wave stages take ~70% of the replay and are the same in every arm.
- Flux where both exact: median 2.7e-11, 99% 2.3e-8, 54 faces above 1e-6: 47 OT (<= 2.3e-6, rotation choice swapped between stages 1 and 2), 6 rotor (2e-4 / 1.6e-6) and one rotor face at 0.18.

**The changed faces, judged by verify_rmhd.py (`planar_ml/extract_face.py`, `face_detail.py`; local re-solve both ways, `planar_ml/face42150/detail.log`):**
- Rotor 42150 (the 0.18): locally production (four-unknown) and the network (late rung, stage 1) give the SAME answer, a weak solution and admissible. The answer production gave in the calea run (harvest row, four-unknown) is a different one and FAILS the verifier at the right fast fan (rh 7e-2). The network fixes this face; production's four-unknown answer is machine-dependent here.
- Rotor 47483 / 47634 (2e-4): production answers with an INTERMEDIATE left slow shock (entropy rising), the network with a left Alfven rotation + regular slow shock. Both pass the verifier; the network gives the regular kind (the 11-05 preference, [[rotation-fused-into-slow-jump]]).
- OT 13439 (2e-6): CORRECTED 11-08 07:40 -- both answers are weak solutions and admissible (fans to 1e-14). The first reading (both fail at the left slow and right fast fans) re-integrated the fans with gamma 5/3: `sol_profiles.py` takes GAMMA from `$SP_POP` at import (default: the 4Q ledger), and verify_rmhd.py uses its integrators. The network's version swaps the right rotation + regular slow shock for an intermediate right slow shock (entropy rising). Verification of every source-2/3 answer of the three harvest runs: 84579 failed (SP_POP unset on calea), rerun as 84580 with each chunk's gamma (`planar_ml/pop_g53.npz`, `pop_g43.npz`).

**Verification of every late-rung / four-unknown answer of the three harvest runs (84580, 11-08 07:45, each run's gamma; `planar_ml/verify_rescues_summary.py`).** 149,130 answers. Weak solutions = admissible everywhere (no entropy-falling shock passes the gate):
- late rung: 4Q 13,956/13,956 (100%), OT 79,826/79,906 (99.90%), rotor 36,349/36,353 (99.99%)
- four-unknown: 4Q 2,240/2,242 (99.91%), OT 3,943/4,058 (97.17%), rotor 12,587/12,615 (99.78%)
- the 229 failures are nearly all the contact out of order with a slow wave (CD FAIL + out of order); OT's late-rung failures are 79 of 80 from stage 3 (pressure-steered)
- INTERMEDIATE slow shocks (speed between the Alfven speeds ahead and behind: the slow shock reverses B_t; entropy rises, so the gate passes it) in 11.9% (4Q) / 20.3% (OT) / 13.1% (rotor) of the late rung's answers -- from the signed-field stages 1-2 (e.g. OT stage 1 no rotation 8,326/33,550; stage 2 with rotation 2,871/8,257), almost never from stage 3 (0/4,734 4Q, 7/7,861 OT, 25/7,095 rotor) or the four-unknown solver (0 / 1.5% / 0.1%). The late rung takes the first branch that converges and passes the gate; it does not prefer the regular branch (rotation + regular slow shock), which production has returned for compound structures since 10-27 ([[compound-selection-is-dissipation]]). The user's call whether the late rung should.

## Resolution ladder (user 11-08: "Now with the fixes it is time to build the resolution ladder. Run 4Q, rotor and OT with 128, 256, 512(except 4Q). The higher resolution only after the lower finished. Give me konservative timings")

`calea_ladder.sh` (PROBLEM, N; worktrees rmhd_final-loose13 + HLLD-fix14): the new chain + kernels.env levers + the late rung with the planar network and regular answers first (loose13 defaults), every face harvested (RMHD_HARVEST_RESCUES, 200k-row shards, no cap), NSNAP 40, networks ft3 (4Q, as its 512^2) / rotor_ft (rotor, OT) + s42/s47. `submit_ladder.sh`: 128^2 now, 256^2 afterok its 128^2, 512^2 (rotor, OT) afterok its 256^2. Output `results/<short>_<N>_exact_ladder`.
- First launch (84590-84599, 08:36) cancelled after ~15 min: the outputs went to HLLD-fix14/results (relative OUT; moved to `*_cancelled1108`) and the rotor's sym rose to 1.1e-3 at step 2.
- The planar network now runs in numpy float64 (`planar_net.forward`, einsum, no BLAS): bitwise batch-independent (20 random batches, single rows), 1.9 ms per 100 faces, within float32 precision of torch (5e-7 relative). Test `test_prediction_does_not_depend_on_the_batch`; test_planar_net.py 9/9.
- Symmetry pilots, rotor 128^2, 3 steps (84600-84602): sym at steps 1/2/3 -- network + regular 1.1e-6 / 1.0e-6 / 4.1e-3; NET=0 REGULAR=0 (production before 11-08) 3.7e-6 / 6.6e-3 / 1.6e-3; regular alone 3.7e-6 / 3.7e-6 / 3.6e-6. NOT the network: production breaks the rotor's symmetry at 128^2 as fast, each jump with an x-sweep face lost (161/162, 169/170) -- one face of a mirror pair solved, its twin on HLLD ([[exact-flux-breaks-symmetry]]); mirror states differ by ~1e-6 after step 1 (solver tolerance), enough to flip a borderline face. The 64^2 runs (1e-8) have no such faces that early.
- Relaunched 08:55: 4Q 84603 (128) -> 84604 (256); rotor 84605 -> 84606 -> 84607 (512); OT 84608 -> 84609 -> 84610 (512). Validation replay of the new defaults 84611 (`calea_replay_net2.sh` netreg, harvested) -> verification 84612 (`calea_verify_pop.sh`: intermediate shocks before/after on the same faces).
- 84611 LOST its results (11-08 12:19/12:46: parts r2d and ot_1 died at the end, after all sweeps): the harvester's close writes a per-call coverage map, `np.stack` of the calls' masks -- equal-length in a run, unequal in a replay -> ValueError; and replay_net.py saved its npz AFTER the close. The answer rows were written (`replay_v3_crashed1108/harvest_*`). Fixed in replay_net.py: the coverage records are dropped after each call, the npz is saved before the close. Cancelled 84611/84612 at 13:0x; rerun (`planar_ml/resubmit_replay_v3.sh`): pilot 85217 (3 sweeps per part) -> netreg 85218 + net 85219 side by side, BOTH harvested on loose13 (net = REGULAR=0: the same faces' answers on the same code without the regular preference; the v2 net arm ran the torch forward of loose12) -> verification 85220 of production's, net's and netreg's answers.

Conservative timings (measured: OT 128^2 2.1 d / 1,321 steps, OT 256^2 5.9 d / 2,699 steps, 4Q 128^2 1.1 d / 738 steps on the October chains; 4Q 512^2 new chain 277 s/step at t ~0.2; per doubling the steps double and the step cost grows 1.4-1.7x; no credit for the new chain's speed-ups, +20-25%): 4Q 128^2 1.5 d, 256^2 4.5 d; OT 2.5 d, 7 d, 25 d; rotor 2.5 d, 7.5 d, 25 d -> 4Q done ~14 Nov, OT and rotor ~12-14 Dec (each 512^2 inside calea's 50-day limit).

4Q 512^2 on the ladder's code (user 11-08: "restart the 4Q 512 run, once it finishes. Then we can also compare"): **84615** `lad-r2d-512` (`calea_ladder.sh` PROBLEM=riemann2d N=512 -> `results/r2d_512_exact_ladder`), from t = 0, dependency `afterany:84516,afterok:84604` -- after the old run finishes AND the ladder's own 4Q 256^2 (the ladder's lower-first rule). The two 512^2 runs then differ only in the chain: 84516 = production's 11-03 chain to step 1380 (77227), loose11 (fixes, no planar net, no regular preference) after; 84615 = loose13 throughout; same networks (ft3 + s42/s47), grid, CFL, NSNAP 40, harvest.
- 84516 at 11-08 10:15: step 1589, t = 0.2133, dt 1.31e-4 -> ~1,425 steps left; new chain 269-296 s/step (steps 1381-1589), the last steps 317-331. The 128^2 production run's step cost grows over t = 0.21-0.4 by 1.13x on average, 1.25-1.45x at the end (`r2d_128_exact_ml_r2d_it3` log) -> ~350 s/step, 5.8 d -> done ~Sat 14 Nov, conservatively (+20%) Sun 15 Nov.
- 84615: ~3,050 steps. Production's 512^2 took 4.4 d to step 1380 (77227: 131 s/step at the start, 372 at 1300); the new chain ran 0.73x production at 512^2 (step 1400) and loose13 0.56x production at 128^2 (steps 0-150: 23/28/45 vs 41/56/80 s/step). Conservative: no credit before step 1380 (4.4 d) + 1,670 steps at ~350 s/step (6.8 d) + 20% = 13.5 d -> starts ~14-15 Nov, done ~29 Nov; likely ~9-10 d (~24 Nov).
- Compare when both are done: exact/lost faces per step and lost-face maps, wall per step, the two runs' difference at the 40 snapshot times (84516's snap_000-018 are in `r2d_512_exact_ml_r2d_it3`, 019+ in `_newchain`), the intermediate-shock share of the late rung's answers (harvests).
