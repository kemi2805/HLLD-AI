"""1D shock tube: `solver: hlld_ai` (one-shot network p*) against `hlld`.

    python scripts/pstar_tube_compare.py --config configs/mignone/st1.yaml \
        --model new_v2=ckpt.pt,stats.npz --model archived=... --out DIR

Runs the 1D driver (src/physics/driver.py, as configured: grid, CFL, t_end,
limiter) once with `hlld` and once per model with `hlld_ai`, and reports
the L1 difference of the final profiles,  sum|q_ai - q_hlld| dx, for rho,
p, v_x, v_y, B_y, plus how often each solver fell back to HLLE (summed over
every flux call of the run).
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

ARCHIVED_HIDDEN = [256, 256, 256, 256, 128, 64]


def counting(fn, tally):
    from src.physics.hlld import LAST_DIAG

    def wrapped(*args, **kw):
        out = fn(*args, **kw)
        tally["calls"] += 1
        tally["faces"] += int(LAST_DIAG.get("n_interfaces", 0))
        tally["hlle_fallback"] += int(LAST_DIAG.get("n_hlle_fallback", 0))
        tally["wave_order_bad"] += int(LAST_DIAG.get("n_wave_order_bad", 0))
        return out
    return wrapped


def run_one(cfg, solver, out_dir, ml=None):
    import src.physics.driver as drv
    c = copy.deepcopy(cfg)
    c["run"]["solver"] = solver
    c["output"]["dir"] = out_dir
    c["output"]["every_n_steps"] = 10 ** 9      # first and last snapshot only
    c["output"].pop("dt_snapshot", None)
    if ml is not None:
        c["ml"] = ml
    tally = dict(calls=0, faces=0, hlle_fallback=0, wave_order_bad=0)
    orig = (drv.hlld_flux, drv.hlld_ai_flux)
    drv.hlld_flux = counting(orig[0], tally)
    drv.hlld_ai_flux = counting(orig[1], tally)
    try:
        t0 = time.time()
        t, grid, prims, _ = drv.run(c)
        wall = time.time() - t0
    finally:
        drv.hlld_flux, drv.hlld_ai_flux = orig
    ph = slice(grid.ng, grid.ng + grid.ncells)
    prof = {k: prims[k][ph].double().numpy() for k in ("rho", "p", "vx", "vy", "By")}
    return t, float(grid.dx), prof, tally, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=CKPT,STATS[,hidden=a-b-c][,act=silu]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.config))[0]
    t_ref, dx, ref, tal, wall = run_one(cfg, "hlld", os.path.join(a.out, f"{tag}_hlld"))
    report = dict(config=a.config, t_end=t_ref, dx=dx,
                  hlld=dict(tally=tal, wall_s=round(wall, 1)), models={})
    print(f"hlld: t={t_ref:.4f} fallbacks {tal['hlle_fallback']}/{tal['faces']} "
          f"({wall:.0f}s)", flush=True)
    for s in a.model:
        name, _, rest = s.partition("=")
        parts = rest.split(",")
        hidden, act = ARCHIVED_HIDDEN, "silu"
        for p in parts[2:]:
            k, _, v = p.partition("=")
            if k == "hidden":
                hidden = [int(x) for x in v.split("-")]
            elif k == "act":
                act = v
        ml = dict(checkpoint=parts[0], norm_stats=parts[1], hidden=hidden,
                  activation=act)
        t, _, prof, tal, wall = run_one(cfg, "hlld_ai",
                                        os.path.join(a.out, f"{tag}_{name}"), ml)
        l1 = {k: float(np.sum(np.abs(prof[k] - ref[k])) * dx) for k in ref}
        l1_rel = {k: float(np.sum(np.abs(prof[k] - ref[k])) /
                           max(np.sum(np.abs(ref[k])), 1e-300)) for k in ref}
        report["models"][name] = dict(t_end=t, L1=l1, L1_rel=l1_rel, tally=tal,
                                      wall_s=round(wall, 1), ml=ml)
        print(f"{name}: t={t:.4f}  L1 rho {l1['rho']:.3e}  p {l1['p']:.3e}  "
              f"vx {l1['vx']:.3e}  By {l1['By']:.3e}  fallbacks "
              f"{tal['hlle_fallback']}/{tal['faces']}  ({wall:.0f}s)", flush=True)
        np.savez(os.path.join(a.out, f"{tag}_{name}_profiles.npz"),
                 **{f"ai_{k}": v for k, v in prof.items()},
                 **{f"hlld_{k}": v for k, v in ref.items()})
    json.dump(report, open(os.path.join(a.out, f"{tag}_compare.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
