#!/usr/bin/env python
"""1D shock-tube reference for replayed faces (planar acceptance study,
2026-11-03): evolve each face as a tube (HLLD + MC + SSP-RK2, src/physics/
tube_seed.run_tubes) to t = 0.45 at several resolutions and read the flux at
x = 0 from the two centre cells (their mean, and their spread as the
uncertainty).  Faces: the flagged ones a configuration (OTHER) left to HLLD,
plus a calibration sample the same configuration answered exactly and
admissibly (the tube error where the exact flux is known).

    HLLD_DIR=<HLLD> python tube_ref.py REPLAY_DIR BASE OTHER OUT.npz [--ncells 1024,2048] [--ncal 200]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("HLLD_DIR", os.getcwd()))


def _tube_chunk(job):
    """One chunk of columns at one resolution, in a worker: the two centre
    cells' primitive states and the final time."""
    import torch
    torch.set_num_threads(1)
    from src.physics.tube_seed import run_tubes
    from src.physics.eos import hybrid_eos
    UL, UR, Bn, N, tend = job
    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    prof, t = run_tubes(UL, UR, Bn, eos, ncells=N, tend=tend, cfl=0.3, limiter="mc", max_steps=8 * N)
    return {c: {k: v[c].detach().cpu().numpy() for k, v in prof.items()} for c in (N // 2 - 1, N // 2)}, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("out")
    ap.add_argument("--ncells", default="1024,2048")
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--tend", type=float, default=0.45)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--max-lost", type=int, default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "64")))
    from src.physics.tube_seed import run_tubes
    from src.physics.eos import hybrid_eos
    from rmhd.batched import jump as J
    A = np.load(os.path.join(a.replay, a.base, "faces.npz"))
    B = np.load(os.path.join(a.replay, a.other, "faces.npz"))
    flag = A["group"] == 0
    lost = np.flatnonzero(flag & A["exact"] & ~B["exact"])[:a.max_lost]
    ok = np.flatnonzero(flag & A["exact"] & B["exact"])
    rng = np.random.default_rng(1103)
    cal = np.sort(rng.choice(ok, min(a.ncal, ok.size), replace=False))
    idx = np.concatenate([lost, cal])
    kind = np.concatenate([np.zeros(lost.size, np.int8), np.ones(cal.size, np.int8)])
    UL, UR = A["U_L"][idx], A["U_R"][idx]
    Bn = UL[:, 7]
    gamma = 5.0 / 3.0
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
    keys = list(A["F_keys"])
    out = dict(idx=idx, kind=kind, F_base=A["F"][idx], F_other=B["F"][idx], F_keys=np.array(keys),
               rung_base=A["rung"][idx], rung_other=B["rung"][idx])
    print("== %d faces %s left to HLLD + %d calibration faces it answered exactly" % (lost.size, a.other, cal.size),
          flush=True)
    for N in [int(x) for x in a.ncells.split(",")]:
        cen = {c: {} for c in (N // 2 - 1, N // 2)}
        jobs = [(UL[c0:c0 + a.chunk, :7], UR[c0:c0 + a.chunk, :7], Bn[c0:c0 + a.chunk], N, a.tend)
                for c0 in range(0, len(idx), a.chunk)]
        import multiprocessing as mp
        import time
        from concurrent.futures import ProcessPoolExecutor
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "60")),
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_tube_chunk, j) for j in jobs]
            done = 0
            results = []
            for f in futs:
                results.append(f.result())
                done += 1
                if done % 20 == 0 or done == len(futs):
                    print("      N=%d  chunk %d / %d  %.0f s" % (N, done, len(futs), time.time() - t0), flush=True)
            tl = []
            for (res, t), job in zip(results, jobs):
                for c in cen:
                    for k, v in res[c].items():
                        cen[c].setdefault(k, []).append(v)
                tl.append(np.full(job[2].shape[0], t))
        t_col = np.concatenate(tl)
        out["t_col_%d" % N] = t_col
        t = float(np.min(t_col))
        print("      N=%d: %d of %d columns reached t_end" % (N, (t_col >= a.tend - 1e-9).sum(), t_col.size),
              flush=True)
        Fs = []
        for c in (N // 2 - 1, N // 2):
            g = {k: np.concatenate(v) for k, v in cen[c].items()}
            v2 = g["vx"] ** 2 + g["vy"] ** 2 + g["vz"] ** 2
            eta = Bn * g["vx"] + g["By"] * g["vy"] + g["Bz"] * g["vz"]
            b2 = (Bn ** 2 + g["By"] ** 2 + g["Bz"] ** 2) * (1.0 - v2) + eta ** 2
            st = [g["rho"], g["p"] + 0.5 * b2, g["vx"], g["vy"], g["vz"], g["By"], g["Bz"]]
            _, F, bad = J.cons_flux(st, Bn, gamma)
            d = {"D": F[0], "Sx": F[1], "Sy": F[2], "Sz": F[3], "tau": F[4] - F[0], "By": F[5], "Bz": F[6],
                 "Bx": np.zeros_like(F[0])}
            Fs.append(np.stack([d[k] for k in keys], axis=1))
        out["F_tube_%d" % N] = 0.5 * (Fs[0] + Fs[1])
        out["F_tube_spread_%d" % N] = np.abs(Fs[0] - Fs[1])
        out["t_%d" % N] = t
        sc = np.max(np.abs(out["F_base"]), axis=1)
        for kd, nm in ((0, "lost"), (1, "calibration")):
            m = (kind == kd) & (t_col >= a.tend - 1e-9) & np.isfinite(out["F_tube_%d" % N]).all(axis=1)
            e_base = np.max(np.abs(out["F_base"] - out["F_tube_%d" % N]), axis=1)[m] / sc[m]
            e_oth = np.max(np.abs(out["F_other"] - out["F_tube_%d" % N]), axis=1)[m] / sc[m]
            q = lambda x: "p50 %.1e p90 %.1e" % tuple(np.quantile(x, [.5, .9])) if x.size else "-"
            print("   N=%d t=%.3f  %-11s |F_%s - tube| %s   |F_%s - tube| %s   %s closer on %d of %d" % (
                N, t, nm, a.base, q(e_base), a.other, q(e_oth), a.base, (e_base < e_oth).sum(), m.sum()), flush=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
