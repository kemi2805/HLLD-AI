# November 2026: the 4Q retrain, the rescue chain, the planar network

These are the scripts and notebooks of the work from 2026-11-01 to
2026-11-08, copied from `claude_scratch/r2d_retrain` on 2026-11-08. Only the
sources are here (.py, .sh, .md, .diff). The results stay on calea under
`/mnt/rafast/miler/codes/rotor2d/HLLD/results/`. The paths in the scripts
point to calea, where they ran through `ssh itp` and sbatch.

The code they ran is committed:

- rmhd, branch `ladder-2026-11`: calea's `rmhd_final-loose11`, `-loose12`
  and `-loose13`, one commit each, on 9ca5a4b.
- HLLD, branch `ladder-2026-11`: calea's `HLLD-fix13` and `HLLD-fix14`, on
  51e9c67.

Every other calea worktree of the month is its base commit plus one of the
patch sets in `fixes/*_patches/`. The `setup_*.sh` there name the recipe.

| Where | What |
|---|---|
| `calea_r2d_it*.sh`, `r2d_*.py` | the 4Q warm-start retrain loop (11-02), which chose ft3 |
| `calea_r2d_audit.sh`, `r2d_audit.py`, `calea_r2d_proper.sh`, `split_proper.py` | the data audit and the proper split (11-03) |
| `drop_inadmissible.py` | inadmissible answers dropped from training only (11-04) |
| `planar_audit/` | the inadmissible slow shocks (11-03); the filter itself is rmhd 627ec1d |
| `anatomy/` | the 4Q 128^2 run face by face (11-04) |
| `fixes/README.md` | the notebook from 11-04 on: loss fixes, gate, rotation re-solve, late rung, cost levers, planar network, resolution ladder |
| `fixes/planar_ml/` | the planar warm-start network: populations, dataset, training (v2a: `calea_planar_train2.sh`, job 84559), replays, verification |
| `fixes/calea_r2d512_new.sh` | the 4Q 512^2 run from step 1380 on the new chain (84516) |
| `fixes/calea_ladder.sh`, `fixes/submit_ladder.sh` | the resolution ladder (84603-84610) and the 4Q 512^2 rerun (84615) |
