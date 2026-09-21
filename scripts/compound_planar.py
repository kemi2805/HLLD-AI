"""Can a REDUCED solver answer the compound-wave interfaces?  (science plan B/C)

The tube test showed the stubborn interfaces reverse B_t inside a jump that
also moves rho and P -- which the jump conditions allow only for an
INTERMEDIATE shock.  Across any discontinuity moving at s, with u = v_x - s,
a^2 = B_x^2 / rho and j = rho u continuous,

    rho_1 (u_1^2 - a_1^2) B_t1  =  rho_2 (u_2^2 - a_2^2) B_t2

so B_t keeps its sign across a fast or a slow shock (both brackets of one
sign) and reverses only across a wave whose flow crosses the Alfven speed.
The production planar solver cannot return one: the slow-shock search in
``rmhd.batched.slow_shock_b`` only looks at speeds up to the Alfven bound
plus ``MARGIN`` = 0.03.

This measures, with NO production change, what each ingredient buys:

  A0  production: the planar solver, its default seed, no flips
  A1  + seeds: default plus B_t = +-|B_t| of either side (no new physics)
  A2  + flips: all four ``flip`` combinations (a pi-rotation before the
      slow wave -- the regular, Lax-admissible way to reverse the field)
  A3  + the intermediate branch: MARGIN widened so the slow-family wave may
      move past the Alfven speed (the compound wave as ONE jump)

Every answer must pass the planar solver's own verification (full
seven-wave residual <= 1e-8).  It is then CLASSIFIED rather than filtered:
is either slow wave intermediate (faster than its Alfven wave), does it flip
B_t, would production's wave-order gate reject it -- and does its structure
agree with the tube's (each non-trivial wave speed matched to a confirmed
tube feature within ``--match``).  Controls are run identically: an answer
that DIFFERS from the one production gives today is the price of the wider
search.

The margin patch reaches only the numpy slow-shock path, so the job must run
with RMHD_SLOWSHOCK unset (enforced below).

    python scripts/compound_planar.py <coplanar_limit dir> <tube screen dir> \
        --shard 0 --nshards 64 --out results/compound_planar/shard_000.npz
    python scripts/compound_planar.py --summarise results/compound_planar
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import subprocess
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
from rmhd.batched import slow_shock_b as SSB                       # noqa: E402
from rmhd.batched import slow_shock_njit as SSN                    # noqa: E402
from src.physics.exact_flux import _self_crossing                  # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

GAMMA = 5.0 / 3.0
ACC = 1e-10
VERIFY = 1e-8
WAVE_ORDER_TOL = 1e-6
FLIPS = ((False, False), (True, False), (False, True), (True, True))
ARMS = ("A0", "A1", "A2", "A3")
GROUPS = ("control", "still-stubborn")
WIDE_MARGIN = 1.0          # A3: the slow-family speed may run to the domain edge


def sub(state, i):
    return [c[i] for c in state]


def load(bdir, sdir, n_control, seed=0):
    B = [np.load(f) for f in sorted(glob.glob(os.path.join(bdir, "shard_*.npz")))]
    c = lambda k: np.concatenate([z[k] for z in B])
    grp, prod, ref = c("group"), c("prod_ok"), c("ref_ok")
    UL, UR = c("UL"), c("UR")
    solv = prod | ref
    still = np.flatnonzero((grp == 1) & ~solv)
    ctrl = np.flatnonzero((grp == 0) & ref)          # production planar answers exist
    if n_control and ctrl.size > n_control:
        ctrl = np.sort(np.random.default_rng(seed).choice(ctrl, n_control,
                                                          replace=False))
    idx = np.concatenate([ctrl, still])
    group = np.concatenate([np.zeros(ctrl.size, np.int8),
                            np.ones(still.size, np.int8)])
    # the tube screen's confirmed features, keyed by the same gidx
    T = [np.load(f) for f in sorted(glob.glob(os.path.join(sdir, "shard_*.npz")))]
    t = lambda k: np.concatenate([z[k] for z in T])
    tg, fsp, fwd, ffam, fk = t("gidx"), t("fspeed"), t("fwidth"), t("ffam"), t("fkind")
    pos = {int(g): i for i, g in enumerate(tg)}
    MAXF = fsp.shape[1]
    tsp = np.full((idx.size, MAXF), np.nan)
    twd = np.full((idx.size, MAXF), np.nan)
    tkind = np.full((idx.size, MAXF), -2, np.int8)
    for r, g in enumerate(idx):
        p = pos.get(int(g))
        if p is not None:
            ok = ffam[p] >= -1
            tsp[r, ok], twd[r, ok], tkind[r, ok] = fsp[p, ok], fwd[p, ok], fk[p, ok]
    return dict(gidx=idx, group=group, UL=UL[idx], UR=UR[idx], tsp=tsp, twd=twd,
                tkind=tkind)


def seeds(L, R):
    """The default planar seed and four B_t seeds (planar frame, signed B_t)."""
    p0 = np.log(np.maximum(0.5 * (L[1] + R[1]), 1e-12))
    out = [None]
    for bt in (np.abs(L[5]), -np.abs(L[5]), np.abs(R[5]), -np.abs(R[5])):
        out.append([p0.copy(), bt.copy(), p0.copy()])
    return out


def structure(F, L, R, Bn, unk3, flip):
    """The equivalent seven-wave answer of a planar one (as verify_full writes
    it): residual, zones, speeds, and the classification."""
    Bt = unk3[1]
    n = Bn.size
    u6 = [unk3[0], np.log(np.maximum(np.abs(Bt), 1e-30)),
          np.where(Bt >= 0.0, 0.0, np.pi), unk3[2],
          np.full(n, np.pi if flip[0] else 0.0), np.full(n, np.pi if flip[1] else 0.0)]
    fv, Z, VL, VR, e = F["fullfuncv6"](L, R, u6, Bn)
    crossed = _self_crossing(Z, VL, VR, WAVE_ORDER_TOL) | e
    # a slow wave ahead of its own Alfven wave is intermediate
    inter = (VL[2] < VL[1] - 1e-9) | (VR[2] > VR[1] + 1e-9)
    # does a slow wave carry B_t through zero?  (planar frame: B_t = By)
    flipL = np.sign(Z[1][5]) * np.sign(Z[2][5]) < 0            # R3 -> R4
    flipR = np.sign(Z[4][5]) * np.sign(Z[3][5]) < 0            # R6 -> R5
    speeds = np.stack([VL[0], VL[2], 0.5 * (Z[2][2] + Z[3][2]), VR[2], VR[0]], 1)
    star = np.stack([Z[2][0], Z[2][1], Z[2][2], Z[3][0]], 1)   # rho4, P*, v*, rho5

    # which of the five waves carries a real jump: a zero-strength wave has no
    # feature in the tube to be matched against
    def jump(a, b):
        return np.maximum.reduce([
            np.abs(a[0] - b[0]) / np.maximum(np.abs(a[0]), 1e-30),
            np.abs(a[1] - b[1]) / np.maximum(np.abs(a[1]), 1e-30),
            np.abs(a[5] - b[5]) / np.maximum(np.abs(a[5]) + np.abs(b[5]), 1e-30),
            np.abs(a[2] - b[2]) + np.abs(a[3] - b[3])])
    live = np.stack([jump(L, Z[0]), jump(Z[1], Z[2]), jump(Z[2], Z[3]),
                     jump(Z[4], Z[3]), jump(Z[5], R)], 1) > 1e-3
    return crossed, inter, flipL | flipR, np.where(live, speeds, np.nan), star


def run(a):
    t0 = time.time()
    tag = "[%s shard %d/%d]" % (socket.gethostname(), a.shard, a.nshards)
    log = lambda s: print("%s %6.0fs  %s" % (tag, time.time() - t0, s), flush=True)
    if SSN.wanted():
        raise SystemExit("RMHD_SLOWSHOCK=njit: the margin patch only reaches the "
                         "numpy slow-shock path -- unset it for this job")
    pop = load(a.lanes, a.screen, a.n_control)
    sel = np.arange(pop["gidx"].size)[a.shard::a.nshards]
    pop = {k: v[sel] for k, v in pop.items()}
    n = sel.size
    log("%d lanes (%d controls, %d still-stubborn)" % (
        n, (pop["group"] == 0).sum(), (pop["group"] == 1).sum()))
    UL, UR = pop["UL"], pop["UR"]
    Bn = UL[:, 7].copy()
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    F = FB.make_solver(GAMMA)
    S5 = P5.make_solver(GAMMA)
    Lp, Rp, _, _ = P5.to_planar(left, right, Bn)

    # per arm: solved?, and the first verified answer's classification
    K = len(ARMS)
    ok = np.zeros((n, K), bool)
    inter = np.zeros((n, K), bool)
    flipb = np.zeros((n, K), bool)
    crossed = np.zeros((n, K), bool)
    how = np.full((n, K, 2), -1, np.int8)            # (flip index, seed index)
    speeds = np.full((n, K, 5), np.nan)
    star = np.full((n, K, 4), np.nan)
    ref_star = np.full((n, 4), np.nan)

    def attempt(k, flip_set, seed_set, margin):
        SSB.MARGIN = margin
        for fi in flip_set:
            for si in seed_set:
                todo = np.flatnonzero(~ok[:, k])
                if todo.size == 0:
                    return
                sd = seeds(sub(Lp, todo), sub(Rp, todo))[si]
                r = S5["solve"](sub(left, todo), sub(right, todo), Bn[todo],
                                unk3_init=sd, accuracy=ACC, flip=FLIPS[fi],
                                verify_tol=VERIFY)
                g = np.flatnonzero(r["converged"])
                if g.size == 0:
                    continue
                lanes = todo[g]
                u3 = [c[g] for c in r["unk"]]
                cr, it, fb, sp, st = structure(F, sub(Lp, lanes), sub(Rp, lanes),
                                               Bn[lanes], u3, FLIPS[fi])
                ok[lanes, k] = True
                crossed[lanes, k], inter[lanes, k], flipb[lanes, k] = cr, it, fb
                speeds[lanes, k], star[lanes, k] = sp, st
                how[lanes, k] = (fi, si)

    attempt(0, (0,), (0,), 0.03)
    # production's own answer on the controls, the reference for "changed?"
    ref_star[:] = star[:, 0]
    log("A0 production: %d/%d" % (ok[:, 0].sum(), n))
    attempt(1, (0,), range(5), 0.03)
    log("A1 + seeds:    %d/%d" % (ok[:, 1].sum(), n))
    attempt(2, range(4), range(5), 0.03)
    log("A2 + flips:    %d/%d" % (ok[:, 2].sum(), n))
    attempt(3, range(4), range(5), WIDE_MARGIN)
    SSB.MARGIN = 0.03
    log("A3 + intermediate branch: %d/%d" % (ok[:, 3].sum(), n))

    # agreement with the tube: each of the answer's five wave speeds against the
    # confirmed features, and every REVERSING tube feature against the answer
    match = np.full((n, K), np.nan)
    rev_on_wave = np.full((n, K), np.nan)
    for j in range(n):
        f = np.isfinite(pop["tsp"][j])
        if not f.any():
            continue
        tsp, twd, tk = pop["tsp"][j, f], pop["twd"][j, f], pop["tkind"][j, f]
        for k in range(K):
            if not ok[j, k]:
                continue
            sp = speeds[j, k][np.isfinite(speeds[j, k])]     # the live waves only
            if sp.size == 0:
                continue
            near = lambda s: np.abs(tsp - s) <= a.match + 0.5 * twd
            match[j, k] = np.mean([near(s).any() for s in sp])
            revs = tsp[tk == 2]
            if revs.size:
                rev_on_wave[j, k] = np.mean(
                    [np.any(np.abs(sp - r) <= a.match + 0.5 * twd.max())
                     for r in revs])
    # the price: on controls, does the wider search change production's answer?
    changed = np.zeros((n, K), bool)
    for k in range(1, K):
        both = ok[:, 0] & ok[:, k]
        d = np.max(np.abs(star[:, k] - ref_star)
                   / (np.abs(ref_star) + 1e-12), axis=1)
        changed[both, k] = d[both] > 1e-6

    np.savez_compressed(
        a.out, gidx=pop["gidx"], group=pop["group"], Bn=Bn, ok=ok, inter=inter,
        flipb=flipb, crossed=crossed, how=how, speeds=speeds, star=star,
        match=match, rev_on_wave=rev_on_wave, changed=changed,
        wall=time.time() - t0, host=socket.gethostname(),
        hlld_rev=subprocess.run(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(),
        argv=" ".join(sys.argv))
    log("wrote %s" % a.out)


def summarise(d):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "shard_*.npz")))]
    if not Z:
        raise SystemExit("no shards in %s" % d)
    c = lambda k: np.concatenate([z[k] for z in Z])
    grp, ok, inter, flipb, crossed = (c("group"), c("ok"), c("inter"), c("flipb"),
                                      c("crossed"))
    match, rev_on, changed = c("match"), c("rev_on_wave"), c("changed")
    print("compound_planar: %d shards, %d lanes (%d controls, %d still-stubborn)"
          % (len(Z), grp.size, (grp == 0).sum(), (grp == 1).sum()))
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        if not m.any():
            continue
        print("\n  %s (n = %d)" % (g, m.sum()))
        print("  %-28s %8s %9s %11s %9s %11s %11s %8s" % (
            "arm", "solved", "new", "intermediate", "flips B_t", "gate rejects",
            "tube match", "changed"))
        prev = np.zeros(m.sum(), bool)
        for k, name in enumerate(("A0 production", "A1 + seeds", "A2 + flips",
                                  "A3 + intermediate branch")):
            o = ok[m, k]
            new = o & ~prev
            mm = match[m, k]
            print("  %-28s %7.1f%% %8d %10d %9d %11d %10s %8s" % (
                name, 100 * o.mean(), new.sum(), (new & inter[m, k]).sum(),
                (new & flipb[m, k]).sum(), (new & crossed[m, k]).sum(),
                ("%.2f" % np.nanmedian(mm[new])) if (new & np.isfinite(mm)).any() else "--",
                ("%d" % changed[m, k].sum()) if k else "--"))
            prev = prev | o
        r = rev_on[m, 3]
        if np.isfinite(r).any():
            print("  on still-stubborn lanes solved by A3 with a reversing tube "
                  "feature: that feature sits on one of the answer's waves in "
                  "%.1f%%" % (100 * np.nanmean(r)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lanes", nargs="?", help="coplanar_limit output directory")
    ap.add_argument("screen", nargs="?", help="tube_features screening directory")
    ap.add_argument("--n-control", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--match", type=float, default=0.03,
                    help="speed tolerance when matching answer waves to tube features")
    ap.add_argument("--out")
    ap.add_argument("--summarise", metavar="DIR")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    if not (a.lanes and a.screen and a.out):
        ap.error("need <lanes> <screen> --out (or --summarise DIR)")
    run(a)


if __name__ == "__main__":
    main()
