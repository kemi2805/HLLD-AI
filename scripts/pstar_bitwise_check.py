"""Is hlld_flux bitwise unchanged?  Main's code against this branch's.

    # 1. inputs, with THIS branch (it has the sampler)
    python scripts/pstar_bitwise_check.py inputs --out in.npz
    # 2. outputs, once per code base -- PYTHONPATH picks which `src` runs
    PYTHONPATH=<main tree>   python <this file> dump in.npz --out main.npz
    PYTHONPATH=<this branch> python <this file> dump in.npz --out br.npz
    PYTHONPATH=<this branch> python <this file> dump in.npz --out brd.npz --row-diag
    # 3.
    python scripts/pstar_bitwise_check.py compare main.npz br.npz brd.npz
    # and a whole 1D run (solver from the config), once per code base:
    PYTHONPATH=<tree> python <this file> tube configs/mignone/st1.yaml --out t.npz

The script deliberately does NOT put its own checkout on sys.path: the
interpreter imports whichever `src` PYTHONPATH names, and `dump` prints it.
Inputs: the union-envelope sampler (both gammas, all four components), the
same rows rotated to idir = 1 and 2, B_n = 0 rows and L = R rows -- every
branch of hlld_flux's fallback logic.  Outputs compared: every flux and
state component and p*, as raw bytes.
"""
import sys

import numpy as np

KEYS = ("rho", "vx", "vy", "vz", "p", "eps", "Bx", "By", "Bz")


def _rot(P, n):
    """Cyclic relabelling x->y->z applied n times to (v, B) of a (N, 9) block."""
    Q = P.copy()
    for base in (1, 6):
        for i in range(3):
            Q[:, base + (i + n) % 3] = P[:, base + i]
    return Q


def make_inputs(out, n=20000, seed=7):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.labelling import make_eos, to_prims
    from src.data.sampling import sample_pairs
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..",
                                           "configs", "pstar-union.yaml")))
    blocks = []
    for gi, g in enumerate((5.0 / 3.0, 4.0 / 3.0)):
        S = sample_pairs(n, cfg["sampling"], np.random.default_rng([seed, gi]), g)
        eos = make_eos(g)
        L, R = to_prims(S["L"], eos), to_prims(S["R"], eos)
        PL = np.stack([L[k].numpy() for k in KEYS], 1)
        PR = np.stack([R[k].numpy() for k in KEYS], 1)
        # degenerate rows: B_n = 0 on 200 rows, L = R on 200 rows
        PL[:200, 6] = PR[:200, 6] = 0.0
        PR[200:400] = PL[200:400]
        for d in (0, 1, 2):
            blocks.append((_rot(PL, d), _rot(PR, d), np.full(n, g), np.full(n, d)))
    PL, PR, G, D = (np.concatenate(x) for x in zip(*blocks))
    np.savez(out, prim_L=PL, prim_R=PR, gamma=G, idir=D.astype(np.int8))
    print(f"{out}: {PL.shape[0]} rows")


def dump(inp, out, row_diag):
    import torch
    torch.set_num_threads(1)
    import src
    from src.physics.eos import hybrid_eos
    from src.physics.hlld import hlld_flux
    print("src =", src.__file__, flush=True)
    z = np.load(inp)
    PL, PR, G, D = z["prim_L"], z["prim_R"], z["gamma"], z["idir"]
    res = {}
    for g in np.unique(G):
        eos = hybrid_eos(K=0.0, gamma=float(g), gamma_th=float(g))
        for d in np.unique(D):
            r = np.flatnonzero((G == g) & (D == d))
            sL = {k: torch.from_numpy(np.ascontiguousarray(PL[r, i])) for i, k in enumerate(KEYS)}
            sR = {k: torch.from_numpy(np.ascontiguousarray(PR[r, i])) for i, k in enumerate(KEYS)}
            kw = dict(row_diag={}) if row_diag else {}
            f, u, p = hlld_flux(sL, sR, eos, idir=int(d), **kw)
            for k in f:
                res.setdefault(f"f_{k}", np.zeros(PL.shape[0]))[r] = f[k].numpy()
                res.setdefault(f"u_{k}", np.zeros(PL.shape[0]))[r] = u[k].numpy()
            res.setdefault("p_star", np.zeros(PL.shape[0]))[r] = p.numpy()
    np.savez(out, src=np.array(src.__file__), **res)
    print(f"{out}: {len(res)} arrays, {PL.shape[0]} rows, "
          f"{int((res['p_star'] < 0).sum())} HLLE fallbacks")


def tube(config, out):
    """A whole 1D run (src/physics/driver.py) with the config's solver."""
    import torch
    import yaml
    import src
    from src.physics.driver import run
    print("src =", src.__file__, flush=True)
    cfg = yaml.safe_load(open(config))
    cfg["output"]["dir"] = out + "_snaps"
    cfg["output"]["every_n_steps"] = 10 ** 9
    cfg["output"].pop("dt_snapshot", None)
    t, grid, prims, cons = run(cfg)
    res = {f"prim_{k}": v.double().numpy() for k, v in prims.items()}
    res.update({f"cons_{k}": v.double().numpy() for k, v in cons.items()})
    res["t"] = np.array([t])
    np.savez(out, src=np.array(src.__file__), **res)
    print(f"{out}: t = {t:.6f}, solver {cfg['run'].get('solver')}")


def compare(ref, *others):
    a = np.load(ref)
    ok = True
    for o in others:
        b = np.load(o)
        keys = sorted(k for k in a.files if k != "src")
        assert keys == sorted(k for k in b.files if k != "src"), (ref, o)
        nbad, worst = 0, 0.0
        for k in keys:
            x, y = a[k], b[k]
            same = (x.view(np.uint64) == y.view(np.uint64))
            nbad += int((~same).sum())
            fin = np.isfinite(x) & np.isfinite(y)
            if fin.any():
                worst = max(worst, float(np.abs(x[fin] - y[fin]).max()))
        print(f"{o} vs {ref}: {len(keys)} arrays x {a[keys[0]].size} rows, "
              f"{nbad} values differ in their bits, max |diff| {worst:.3e}  "
              f"[{str(a['src'])} | {str(b['src'])}]")
        ok &= nbad == 0
    print("BITWISE IDENTICAL" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inputs":
        make_inputs(sys.argv[sys.argv.index("--out") + 1])
    elif cmd == "dump":
        dump(sys.argv[2], sys.argv[sys.argv.index("--out") + 1],
             "--row-diag" in sys.argv)
    elif cmd == "tube":
        tube(sys.argv[2], sys.argv[sys.argv.index("--out") + 1])
    elif cmd == "compare":
        sys.exit(compare(*sys.argv[2:]))
    else:
        sys.exit(f"unknown command {cmd}")
