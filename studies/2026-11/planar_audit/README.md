# Planar acceptance / inadmissible slow shocks (2026-11-03)

Question (user): the 4Q 64^2 data audit flagged 37 harvested problems whose
stored solution has seven-wave residual > 1e-6 (up to 5.3), mostly planar
five-wave rescues. What does production's planar acceptance check, are these
answers admissible, how many faces used them, what is the fix?

Copies of the scripts live on calea in /mnt/rafast/miler/scratch/r2d_retrain/planar_audit;
outputs in /mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/planar_audit.

## Scripts
- `wavecheck.py` -- wave-by-wave check of stored rows: RH residual per wave
  (planar4-style scaled certificate), Lax margins per family, entropy ratio
  K = p/rho^gamma (`admissibility`), fan head/tail, contact jumps.
- `extract_rows.py` / `calea_extract.sh` (job 77246) -- the 39 flagged rows +
  3000 seven-wave / 3000 planar control rows -> `rows_1103.npz`.
- `run_check1..3.py`, `control_breakdown.py`, `characterise.py`, `laxonly.py` -- local analyses.
- `repro_planar.py`, `trace_frames.py`, `batchdep.py`, `demo_root.py` -- the mechanism:
  verify_full passes in the solver's own inputs; an ulp change of the inputs
  switches the slow-shock sub-solver between an expansion root and a
  compressive root (not a frame effect, not batch dependence).
- `alt_solution.py` (+ `alt_solution_lib.py`) -- planar5 with slow_shock_b.MARGIN = 0;
  `alt_margin0.npz`.
- `admiss_all.py` / `calea_admiss.sh` (job 77249) -- flags over it1/all.npz + it2/perturb.npz
  -> `admiss_1103.json`, `admiss_flagged_1103.npz`.
- `admiss_harvest.py` / `calea_admiss_harvest.sh` (job 77252) -- the same over six harvests
  (4Q/OT/rotor 64^2, alfx1 and ssretry) -> `admiss_harvests_1103.json`.
- Patches (UNCOMMITTED, opt-in, off = production bitwise):
  - `exact_flux_admissible.diff` (HLLD 51e9c67): `_inadmissible()` + RMHD_ADMISSIBLE=report|1
    at the seven-wave stage, `_planar_and_ray` (reason 8), `_planar4_and_ray` (reason 9);
    diag n_inadmissible7, inadmissible7/5/4 counts.
  - `slow_shock_admissible.diff` (rmhd_final 9ca5a4b): RMHD_SLOWSHOCK_ADMISSIBLE=1, numpy
    ladder only -- candidates ranked only if entropy does not fall (1e-6) and the slow
    family's Lax conditions hold (1e-4).
  - applied in calea worktrees rotor2d/HLLD-admiss and rotor2d/rmhd_final-admiss
    (`setup_calea_worktrees.sh`); local copies HLLD/.claude/worktrees/admiss and this
    session's rmhd_final worktree.
- `r2d_admiss_replay.py` / `calea_admiss_replay_{njit,numpy}.sh` (jobs 77250, 77251) --
  replay 2,525 flagged problems + 5,000 controls under configurations
  A_report, B_refuse, Anp_report, C_filter, CB_filter_refuse -> replay/<cfg>/faces.npz.
  `compare_replay.py replay BASE OTHER` compares; `fetch_replay.sh` copies.
- `tube_ref.py` / `calea_tube_ref.sh` (job 77253) -- 1D tube reference for the faces B
  left to HLLD + 200 calibration faces.

## Results (see memory note inadmissible-slow-shocks for the full table)
- Production planar acceptance = planar5_b.solve converged (3x3 Newton) AND planar AND
  verify_full <= 1e-8 (full seven-wave residual, planar frame) + ray + physical; wave
  order only on flip branches; `verified |= take5`.
- Flagged answers: left fast shock genuine; left slow "shock" an expansion shock riding
  on it beyond the Alfven speed (entropy x0.89, rho x0.32), RH-exact.
- 4Q 64^2: 5,114 exact faces (0.48%) flagged, 4,222 entropy-falling; OT/rotor similar,
  rotor planar answers 4.9%. Not caused by the slow-shock retry.
- Flux: admissible answer exists for ~90%; production's flux within p50 ~1e-5.
  Refuse (B) loses 574/2525 to HLLD; root filter (C) loses 238 and keeps 1,465 on the
  seven-wave path.

## njit port and paired 64^2 runs (2026-11-03, afternoon)
- The filter is ported to `slow_shock_njit` (same switch); new test
  `tests/batched/test_slow_shock_admissible.py` (rmhd worktree, uncommitted;
  `test_slow_shock_admissible.diff`). Calea worktree updated by `update_rmhd_worktree.sh`.
- `calea_adm_validate.sh` (job 77964): 34 tests pass; njit-filter replay == numpy-filter replay
  bitwise on 7,525 faces.
- `calea_adm64.sh` (jobs 77965 r2d, 77966 ot, 77967 rotor): switches-off pilot (matches the
  control bitwise), then the treatment run -> results/{r2d,ot,rotor}_64_exact_admiss.
- `calea_adm64_post.sh` (job 77970, after the three): harvest admissibility audit + face counts.
- When done: `./fetch_adm64.sh`, then per problem
  `python adm_compare64.py cmp_<p> ctl=runs_adm64/<p>_ctl adm=runs_adm64/<p>_adm --ref runs_adm64/ref_<p>.npz --base ctl --tend <0.4|1.0> --title <...>`.
- Interim (runs still going): `interim_adm64.py runs_adm64` (counts over the steps both runs
  closed), `interim_where.py runs_adm64 interim_1103` (per-face lost difference + map),
  `interim_rungs.py runs_adm64` (what each run answered the changed faces with),
  `rotor_step1/match_lost.py` (bitwise-identical problems of the first rotor shard).
- 4Q done 19:54: `cmp_r2d/` (summary.json, lost_maps.png, error_maps.png, wall.png);
  `calea_adm64_audit_r2d.sh` (job 78280) -> admiss_harvests_adm64_r2d.json.
  `wait_state.sh JOBID [HOURS]` = sacct-based watcher (ssh failures retried).

## All three pairs done; the loose-fallback leak (2026-11-04, night)
- OT done 02:25, rotor 23:35, post audit 77970 -> admiss_harvests_adm64.json.
  `fetch_adm64.sh ot_adm:... rotor_adm:...`; `cmp_ot/`, `cmp_rotor/` (adm_compare64.py,
  `--tend 1.0` / `0.4`); `where_1104/` (interim_where.py on the full runs).
- Physical faces lost, control -> filter: 4Q 7,758 -> 8,164 (+5.2%), OT 25,782 -> 26,718
  (+3.6%), rotor 12,622 -> 11,349 (-10.1%, trajectory: +422/+100/-337/-1,458 by window,
  417 faces worse / 405 better). L1(rho) vs 512^2 unchanged in all three. Wall +5% / 0% / -11%
  (different nodes). Rotor sym_err at t=0.4 1.5e-3 -> 3.0e-3.
- Remaining flags: `extract_flagged.py` / `calea_extract_flagged.sh` (job 78316) ->
  flagged_adm64/ (flagged rows + per-wave diagnostics); `extract_rh_tail.py` /
  `calea_extract_rh_tail.sh` (job 78317) -> rh_tail_adm64/ (RH-certificate histogram of every
  accepted shock, rows above 1e-5). `rotor_step1/loose_check.py`, `resolve_tail.py` (re-solve a
  stored slow shock alone, filter off/on: REPRO / ROOT / FAIL), `classify_tail.py` (the
  sub-solver's own residual max(|f_hug|,|f_416|)/(1+|P_a|) at the stored shock: root <= 1e-8,
  loose > 1e-6, nothing between).
- Result: every remaining flagged row of the filter runs is the ladder's LOOSE FALLBACK (a grid
  point, RH certificate 1e-3..1e-1) accepted as exact -- except the rotor's 108 planar5
  fast-shock Lax marginals (unfiltered by design). Loose rows accepted: production 0 / 43 / 17,
  filter 92 / 1,887 / 557 (4Q / OT / rotor). Without the filter the same inputs have a root whose
  entropy falls by ~1e-6..2e-5. The outer residual carries only eq. 4.17 of a slow shock (zero in
  the plane), so neither the seven-wave Newton nor planar5's verify_full can see a loose cell.
  OT's 32k slow shocks with certificate 1e-5..1e-3 (both runs) are converged roots (outer slack
  ~5e-9), not loose.

## The loose-fallback fix and the reruns (2026-11-04, morning; user: "Yes, fix and rerun")
- Fix (option 1 of the two offered): under RMHD_SLOWSHOCK_ADMISSIBLE=1 the ladder withholds its
  loose fallback on every lane where the filter discarded a converged root (numpy `attempt`
  returns `dropped`, `ladder` keeps `discarded`; njit `attempt_s` / `solve_lanes` the same).
  Those lanes FAIL and go to the retry. With the switch off `discarded` is always False: old code.
  Production-type loose lanes (no converged root at all) are untouched, so the filter-on loose
  answers are now EXACTLY the filter-off ones, bitwise (fix_loose/inv_check.py, seeds 99/7/3,
  LS/RS, retry on/off).
- Laptop: fix_loose/fixcheck_loose.py on 40-lane samples of every run's leaked shocks: filter on
  -> FAIL on all (ladder only and with the retry), numpy == njit counts; filter off -> a root
  (1-K p50 4e-7..9e-6), except 6 production-type REPRO lanes. Tests: test_slow_shock_admissible.py
  12 passed (6 new: the 4Q leak lane shard 33 row 10, x2 retry; the bitwise loose-cell invariant,
  LS/RS x retry), slow-shock njit/retry/shock suites 28 passed.
- calea: patch fix_loose/slow_shock_admissible_1104.diff + test_..._1104.diff on rmhd_final-admiss
  (update_rmhd_worktree_1104.sh; md5 == laptop). Jobs (submit_admfix.sh): validate 78332 (all
  leaked shocks re-solved, the 7,525-face replay numpy vs njit bitwise + vs the pre-fix replays,
  the full rmhd suite), reruns 78333 r2d / 78334 ot / 78335 rotor (calea_admfix64.sh: switches-off
  pilot vs control, then RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report; outputs
  results/{r2d,ot,rotor}_64_exact_admfix), post 78336 afterany (admiss_harvest -> 
  admiss_harvests_admfix.json, flagged_admfix/, rh_tail_admfix/, run_meta table).
- 09:05-09:08: all three pilots (switches off, 4 steps) MATCH their controls bitwise; treatments
  started. Validate 78332 parts 1-2 (log slurm_admfix-validate_78332.out):
  (1) EVERY leaked slow shock of the first filter runs re-solved (numpy == njit, same counts):
      filter on now FAILS on all of them, ladder and retry alike -- 4Q 98/98, OT 1,849/1,889,
      rotor 631/655; the rest (OT 40, rotor 24) are production-type: no root even unfiltered
      (REPRO off and on). Unfiltered roots: entropy falls > 1e-6 on 4Q 68/98, rotor 339/631,
      OT only 278/1,849 -- OT's leak was mostly the LAX test (entropy rising, 1-K p50 -1.8e-5 on LS).
  (2) 7,525-face replay with the fix: numpy == njit bitwise (7,525/7,525, exact, rung); acceptance
      counters all 0 inadmissible; vs the pre-fix filter: exact 7,284 -> 7,283, 1,007 fluxes moved
      (fix_loose/replay: 986 still exact, |dF|/|F| p50 9e-12 p90 4e-10 p99 2e-4 max 1e-2; 11
      exact -> not, 10 not -> exact); controls vs production 4,573/4,999 bitwise, max 3.2e-6
      (as before). Wall 185 s njit (pre-fix 183 s).
  (3) rmhd suite, clean env, 64 workers (11:00, log logs/admfix_pytest_78332.log): 217 passed
      (205 + the 12 of test_slow_shock_admissible.py), 3 skipped, 3 failed = calea's known three,
      the two deterministic ones with the SAME values as the ssretry and slowfan full gates
      (Alfven LA 0.018142771445763678; warm start "5 more lanes"); the third is the timing ratio.
- Early look (step 100, t=0.138): rotor sym 6.6e-4 (control 2.0e-3, round 1 7.5e-4); round 1
  doubled only later (3.0e-3 at t=0.4), so this decides nothing. Wall at step 100: 6,788 s
  (control 5,701 s, round 1 6,303 s; different nodes).
- Rotor sym is NOISY step to step: control alone 1.48e-3 / 2.78e-3 / 4.96e-3 / 2.86e-3 / 1.53e-3 at
  steps 150/200/230/256/284; round 1 8.4e-4 / 3.0e-3 / 2.0e-3 / 2.5e-3 / 3.0e-3; rerun 4.2e-4 /
  3.2e-3 / 3.8e-3 / 1.5e-3 at 150-256. The round-1 "1.5e-3 -> 3.0e-3 at t=0.4" is one sample of
  this scatter, NOT a filter effect -- do not quote it.
- 4Q RERUN 78333 DONE 17:51 (calea05; compare_fix.sh r2d 78333 -> cmp_fix_r2d/, where_fix_r2d/,
  where_fix_r2d_vs_adm/). Physical faces lost: control 7,758 / round 1 8,164 (+5.2%) / fix 8,024
  (+3.4% vs control, -140 vs round 1). L1(rho) vs 512^2: 0.1022505 / 0.1022612 / 0.1022518; max
  |drho| fix - control 2.8e-3. Wall 29,742 / 31,211 / 31,506 s (+4.9% / +5.9%, different nodes).
  Rungs (physical): seven 887,222 / 891,398 / 889,982, planar5 135,672 / 132,382 / 133,926, planar4
  3,966 / 2,562 / 2,574. WHERE: vs control the extra losses sit on a handful of faces in the
  lower-left roll-up between the shear layers, t > 0.2 (top 10 faces carry +400 of the +588 gross
  increase; mirror pairs); vs round 1 the fix wins back faces along the NE diagonal jet. Flags
  (admiss_harvest) come with post 78336.
- User decision (relayed by exact Riemann, 11-04): drop inadmissible rows from TRAINING only
  (their job 78340, *_adm.npz; VAL/TEST untouched); in the RUN keep inadmissible exact answers
  where they beat HLLD. exact Riemann proposed a lane-level preference (not approved). My reply:
  the rule is per face -> veto + fix + a final filter-off pass on the faces still lost; a lane-level
  preference would answer the 217 replay faces the fix moved seven -> planar5/planar4 (all
  admissible) with inadmissible seven-wave roots. Evidence "beats HLLD" is 4Q only (tube 77311).

## Fix reruns: final (2026-11-05; post 78336 done 01:38)
compare_fix.sh {r2d 78333, ot 78334, rotor 78335} -> cmp_fix_<p>/ (lost_maps, error_maps, wall),
where_fix_<p>/ (vs control), where_fix_<p>_vs_adm/ (vs round 1); flags admiss_harvests_admfix.json,
flagged_admfix/, rh_tail_admfix/ (classify_tail.py, resolve_tail.py 1e-3 1e3).

| ctl / round 1 / fix      | 4Q                        | OT                          | rotor                        |
|--------------------------|---------------------------|-----------------------------|------------------------------|
| physical faces lost      | 7,758 / 8,164 / 8,024     | 25,782 / 26,718 / 27,607    | 12,622 / 11,349 / 11,485     |
|   vs control             | +5.2% / +3.4%             | +3.6% / +7.1%               | -10.1% / -9.0% (trajectory)  |
| L1(rho) vs 512^2         | .1022505/.1022612/.1022518| .2656772/.2656714/.2656803  | .2873735/.2874126/.2874299   |
| wall (s; nodes differ)   | 29,742 / 31,211 / 31,506  | 54,568 / 54,576 / 59,416    | 49,719 / 44,261 / 38,002     |
| flagged harvest rows     | 5,114 / 92 / 2            | 16,526 / 1,803 / 32         | 14,395 / 665 / 129           |

- Fix runs: ZERO entropy-falling rows anywhere. Remaining flags = (a) loose fallbacks 2 / 30 / 25,
  ALL production-type (resolve_tail: REPRO with the filter off AND on -- no root even unfiltered;
  production had 0 / 43 / 17), (b) planar5 fast-shock Lax marginals 0 / 2 / 104 (unfiltered by
  design). Every other rh > 1e-5 slow shock is a converged root (OT 33,040; |eq417| p50 2-5e-9).
- Rotor by window (t 0-0.1 / 0.1-0.2 / 0.2-0.3 / 0.3-0.4) fix - control: +464 / +167 / -275 /
  -1,493. Early, where states are still close, the filter loses 43% more faces (1,077 -> 1,541;
  round 1 1,499); the overall -9% is trajectory.
- OT vs round 1 +889 physical (+1,402 all faces): about what round 1's 1,844 leaked rows predict if
  no lower rung rescues them; spread thinly over the current sheets (top 50 faces carry 491 of the
  +1,493 gross). Max |drho| fix - control 2.4e-3 (OT), 2.8e-3 (4Q).
- 4Q vs round 1 -140: trajectory (fixed-state replay 7,284 -> 7,283).
- Rotor sym, geometric mean over t > 0.2: 2.50e-3 / 2.33e-3 / 1.82e-3 (max 9.4e-3 / 1.4e-2 /
  1.5e-2). No systematic change.
- Verdict: the fix does what it should (no filter-caused leak survives); the filter costs +3-7%
  lost faces (4Q, OT) and ~6-9% wall for no L1 change. Under the user's rule (keep inadmissible
  answers in the run where they beat HLLD) the run stays on production; the per-face fallback
  would only make answers admissible where possible, with no measurable L1 effect at 64^2.
