"""Rotor 64^2, first harvest shard (steps 1-10) of the control (production)
and the filter run: problems with bitwise-equal (U_L, U_R) that one run
solved and the other lost.  For the ones the filter lost, was the control's
answer admissible (wavecheck.admissibility: entropy of every shock, Lax)?

    PYTHONPATH=<rmhd_final> python match_lost.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavecheck as WC  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))


def keyed(f):
    z = np.load(os.path.join(H, f), allow_pickle=True)
    keys = [a.tobytes() + b.tobytes() for a, b in zip(z["U_L"], z["U_R"])]
    return z, {k: i for i, k in enumerate(keys)}


def main():
    cs, cs_k = keyed("ctl_solved_0.npz")
    cu, cu_k = keyed("ctl_unsolved_0.npz")
    ts, ts_k = keyed("adm_solved_0.npz")
    tu, tu_k = keyed("adm_unsolved_0.npz")
    print("control: %d solved, %d unsolved rows; filter: %d solved, %d unsolved" % (
        len(cs_k), len(cu_k), len(ts_k), len(tu_k)))
    both = [k for k in tu_k if k in cs_k]                    # filter lost, control solved
    back = [k for k in ts_k if k in cu_k]                    # filter solved, control lost
    same_lost = [k for k in tu_k if k in cu_k]
    print("same problem: lost by both %d; lost only by the filter %d; solved only by the filter %d" % (
        len(same_lost), len(both), len(back)))
    for name, ks, Z, Zk, U, Uk in (("lost only by the filter: CONTROL's answer", both, cs, cs_k, tu, tu_k),
                                   ("solved only by the filter: FILTER's answer", back, ts, ts_k, cu, cu_k)):
        if not ks:
            continue
        i = np.array([Zk[k] for k in ks])
        j = np.array([Uk[k] for k in ks])
        g = float(Z["gamma"])
        d = WC.admissibility(Z["U_L"][i], Z["U_R"][i], Z["zones"][i], Z["speeds"][i], g)
        Kmin = np.ones(i.size)
        lax = np.full(i.size, np.inf)
        for w in ("LF", "LS", "RS", "RF"):
            sh = d[w + "_solver_kind"] == 1
            Kmin = np.where(sh, np.minimum(Kmin, d[w + "_K"]), Kmin)
            if w + "_lax_min" in d:
                lax = np.where(sh, np.minimum(lax, d[w + "_lax_min"]), lax)
        src = Z["source"][i]
        print("== %s (%d faces)" % (name, i.size))
        print("   source (0 seven-wave, 1 planar5, 2 planar4?): %s" % dict(zip(*np.unique(src, return_counts=True))))
        print("   entropy falls (min K_b/K_a < 1-1e-6): %d   min K: %s" % (
            int((Kmin < 1 - 1e-6).sum()), np.round(np.sort(Kmin)[:8], 4)))
        if np.isfinite(lax).any():
            print("   Lax margin < -1e-4: %d" % int((lax < -1e-4).sum()))
        if "reason" in U.files:
            print("   the other run's refusal reason: %s" % dict(zip(*np.unique(U["reason"][j], return_counts=True))))
        print("   attempts: %s" % dict(zip(*np.unique(Z["attempts"][i], return_counts=True))))
    print("admissibility keys:", sorted(k for k in d if k.endswith("_lax") or k.endswith("_K"))[:12])


if __name__ == "__main__":
    main()
