#!/usr/bin/env python3
"""Balsara test 1 (relativistic Brio-Wu, Gamma = 2): what our exact solver returns, how
it was checked, and what finite-volume runs converge to.

The inputs are the npz files ``scripts/compound_wave.py --balsara1`` writes: the exact
answers (``regular_flip00_*``: the planar root production finds; ``compound_*``: the
compound alternative) and the tubes (``tube_<limiter>_<n>_<var>``, or ``tube_<n>_<var>``
in the oldest file, which ran mc).  The file does not record the flux, so each run is
named on the command line as FLUX:FILE.

    python scripts/balsara1_study.py checks  --run hlld:<dir>/b1_hlld_mc.npz --run hlle:...
    python scripts/balsara1_study.py figures --run ... --full hlld:mc:512 \\
        --zoom hlld:mc:1024 hlld:mc:4096 hlle:mc:4096 --out docs/figs/balsara1
    python scripts/balsara1_study.py roots 00 10 01 11        # on calea: kernels.env, one process per setting

checks   our answer, independently: the jump conditions across each discontinuity
         recomputed with HLLD's flux formulas (src/physics/hlld.py shares no code with
         the exact solver), the entropy p/rho^Gamma across each shock, the left slow
         wave's speed against the characteristic speeds on both sides; then every run's
         L1 against our answer (overall and away from the discontinuities) and its
         mean deviation from our plateau just behind the left slow wave, where the
         compound answer's attached fan would sit.
figures  <out>.png: the profiles with the zooms; <out>_fan.png: the left slow wave on
         its own, our single intermediate shock against the compound answer, whose fan
         is integrated ray by ray and checked (each state moves at its own ray's slow
         speed; it starts sonic behind its shock; it ends on the contact state).
roots    every planar root reachable from a 224-start grid, with and without a
         pi-rotation at either Alfven wave (flip settings 00, 10, 01, 11).

Needs matplotlib for `figures` (the laptop; calea's venv has none).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmhd.eos import set_eos                                      # noqa: E402
set_eos("ideal")
from rmhd.initial_data import load_initial_data                   # noqa: E402
from rmhd.batched import profile as PR                            # noqa: E402
from rmhd.batched import ray_b as RAY                             # noqa: E402
from rmhd.batched import wave_speeds_b as WB                      # noqa: E402

T = 0.4
VARS = ("rho", "p", "vx", "vy", "By")
Bx, G, LEFT, RIGHT = load_initial_data(8)


# ── the two exact answers ────────────────────────────────────────────────────

def answers(f):
    """Our planar root (zones Z0..Z3, speeds LF LS CD RS RF) and the compound states."""
    z = np.load(f)
    return dict(Z=z["regular_flip00_zones"], sp=[float(v) for v in z["regular_flip00_speeds"]],
                A=z["compound_A"], S=z["compound_S"], B=z["compound_B"],
                Vsh=float(z["compound_Vshock"]), Vtl=float(z["compound_Vtail"]))


def prim(st):
    d = PR.prim_from_seven([np.array([float(c)]) for c in st], np.array([float(Bx)]))
    return {k: float(np.asarray(v).ravel()[0]) for k, v in d.items()}


def ours_at(ans, xi, sampler=None):
    """Our answer read along the rays: the planar root as a seven-wave profile whose
    Alfven waves carry nothing (placed beside the slow waves)."""
    Z, sp = ans["Z"], ans["sp"]
    zones6 = [list(Z[0]), list(Z[0]), list(Z[1]), list(Z[2]), list(Z[3]), list(Z[3])]
    VsL = [sp[0], sp[1] - 1e-9, sp[1]]
    VsR = [sp[4], sp[3] + 1e-9, sp[3]]
    ex, _reg, err = PR.profile_at(LEFT, RIGHT, Bx, zones6, VsL, VsR, G, np.asarray(xi),
                                  sampler or PR.make_sampler(G))
    if np.any(err):
        raise RuntimeError("%d rays failed" % int(np.count_nonzero(err)))
    return ex


class Ours:
    """Our answer at any rays, fast: the constant states exactly, the two fast fans
    interpolated from tables integrated once (``ours_at`` integrates every ray, which on
    a 32768-cell run is most of the cost).  The tables are 2000 rays each; the fans are
    smooth: linear interpolation agrees with ray-by-ray integration to 1.3e-7
    (measured on 3001 rays, printed by `checks`)."""

    def __init__(self, ans, n_table=2000, sampler=None):
        sampler = sampler or PR.make_sampler(G)
        Z, sp = ans["Z"], ans["sp"]
        e0 = WB.xi_all(*[np.array([float(c)]) for c in Z[0]], np.array([float(Bx)]), G)[0][0]
        e3 = WB.xi_all(*[np.array([float(c)]) for c in Z[3]], np.array([float(Bx)]), G)[0][0]
        self.lf = (sp[0], float(e0[0]))             # head, tail of the left fast fan
        self.rf = (float(e3[3]), sp[4])             # tail, head of the right fast fan
        self.sp = sp
        self.const = [prim(LEFT), prim(Z[0]), prim(Z[1]), prim(Z[2]), prim(Z[3]), prim(RIGHT)]
        self.tables = []
        for lo, hi in (self.lf, self.rf):
            xs = np.linspace(lo, hi, n_table)
            self.tables.append((xs, ours_at(ans, xs, sampler)))

    def __call__(self, xi):
        xi = np.asarray(xi, dtype=float)
        sp = self.sp
        edges = [self.lf[0], self.lf[1], sp[1], sp[2], sp[3], self.rf[0], self.rf[1]]
        out = {}
        for v in VARS + ("vz", "Bz"):
            if v not in self.const[0]:
                continue
            q = np.empty_like(xi)
            # L | LF fan | Z0 | Z1 | Z2 | Z3 | RF fan | R
            q[xi < edges[0]] = self.const[0][v]
            q[(xi >= edges[1]) & (xi < edges[2])] = self.const[1][v]
            q[(xi >= edges[2]) & (xi < edges[3])] = self.const[2][v]
            q[(xi >= edges[3]) & (xi < edges[4])] = self.const[3][v]
            q[(xi >= edges[4]) & (xi < edges[5])] = self.const[4][v]
            q[xi >= edges[6]] = self.const[5][v]
            for (xs, tab), m in zip(self.tables, ((xi >= edges[0]) & (xi < edges[1]),
                                                  (xi >= edges[5]) & (xi < edges[6]))):
                q[m] = np.interp(xi[m], xs, tab[v])
            out[v] = q
        return out


def compound_fan(ans, xi, sampler=None):
    """The compound answer's attached slow fan at rays ``xi`` (inside [Vshock, Vtail])
    and three checks of it."""
    xi_fn, _fan_p, fan_n = sampler or PR.make_sampler(G)
    S, B = ans["S"], ans["B"]
    i = np.arange(np.size(xi))
    st, err = RAY._fan_state([np.full(i.size, float(c)) for c in S], float(np.hypot(B[5], B[6])),
                             "LS", np.full(i.size, float(Bx)), i, np.asarray(xi), G, xi_fn,
                             fan_n, 60, False)
    eig = WB.xi_all(*st, np.full(i.size, float(Bx)), G)[0]
    eS = WB.xi_all(*[np.array([float(c)]) for c in S], np.array([float(Bx)]), G)[0][0]
    checks = dict(failed=int(np.count_nonzero(err)),
                  ray_vs_slow_speed=float(np.max(np.abs(eig[:, 1] - np.asarray(xi)))),
                  sonic_head=abs(float(eS[1]) - ans["Vsh"]),
                  tail_vs_B=max(abs(float(st[k][-1]) - float(B[k])) for k in range(7)))
    return PR.prim_from_seven(st, np.full(i.size, float(Bx))), checks


def alfven(q):
    vx, vy, vz, By, Bz = q["vx"], q["vy"], q["vz"], q["By"], q["Bz"]
    W = 1.0 / np.sqrt(1.0 - vx * vx - vy * vy - vz * vz)
    vB = vx * Bx + vy * By + vz * Bz
    b0 = W * vB
    bx = Bx / W + b0 * vx
    b2 = (Bx * Bx + By * By + Bz * Bz) / W ** 2 + vB * vB
    sw = np.sqrt(q["rho"] * (1.0 + G / (G - 1.0) * q["p"] / q["rho"]) + b2)
    return sorted([(bx + sw * W * vx) / (b0 + sw * W), (bx - sw * W * vx) / (b0 - sw * W)])


# ── the runs ─────────────────────────────────────────────────────────────────

def load_runs(specs):
    """{(flux, limiter, n): dict(x, rho, p, vx, vy, By)} from FLUX:FILE specs."""
    runs = {}
    for spec in specs:
        flux, f = spec.split(":", 1)
        z = np.load(f)
        for k in z.files:
            m = re.fullmatch(r"tube_(?:([a-z0-9]+)_)?(\d+)_xi", k)
            if not m:
                continue
            lim, n = m.group(1) or "mc", int(m.group(2))
            pre = k[:-2]
            runs[(flux, lim, n)] = dict(xi=z[k], x=0.5 + T * z[k],
                                        **{v: z[pre + v] for v in VARS})
    return runs


def run_key(s):
    flux, lim, n = s.split(":")
    return flux, lim, int(n)


# ── checks ───────────────────────────────────────────────────────────────────

def checks(a):
    import torch
    from src.physics.eos import hybrid_eos
    from src.physics.hlld import compute_srmhd_fluxes
    ans = answers(a.answers)
    Z, sp = ans["Z"], ans["sp"]
    eos = hybrid_eos(K=0.0, gamma=G, gamma_th=G)
    P = [prim(s) for s in (LEFT, Z[0], Z[1], Z[2], Z[3], RIGHT)]

    def tor(q):
        d = {k: torch.tensor([v], dtype=torch.float64) for k, v in q.items()}
        d["Bx"] = torch.tensor([float(Bx)], dtype=torch.float64)
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    print("our answer: LF %+.5f  LS %+.5f  CD %+.5f  RS %+.5f  RF %+.5f" % tuple(sp))
    keys = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
    for i, j, s, name in ((1, 2, sp[1], "left slow wave"), (2, 3, sp[2], "contact"),
                          (3, 4, sp[3], "right slow shock")):
        uL, uR, fL, fR, _, _ = compute_srmhd_fluxes(tor(P[i]), tor(P[j]), eos, idir=0)
        sc = max(max(abs(float(fL[k])), abs(float(fR[k]))) for k in keys)
        res = max(abs(float(fR[k] - fL[k] - s * (uR[k] - uL[k]))) for k in keys) / sc
        line = "  %-17s jump conditions %.1e" % (name, res)
        if name != "contact":                       # no gas crosses a contact
            up, dn = (i, j) if P[i]["vx"] > s else (j, i)
            line += "   entropy p/rho^G %.5f -> %.5f" % (P[up]["p"] / P[up]["rho"] ** G,
                                                         P[dn]["p"] / P[dn]["rho"] ** G)
        print(line)
    e0 = WB.xi_all(*[np.array([float(c)]) for c in Z[0]], np.array([float(Bx)]), G)[0][0]
    e1 = WB.xi_all(*[np.array([float(c)]) for c in Z[1]], np.array([float(Bx)]), G)[0][0]
    print("  left slow wave %+.5f: ahead A- %+.5f s- %+.5f, behind A- %+.5f s- %+.5f"
          % (sp[1], alfven(P[1])[0], e0[1], alfven(P[2])[0], e1[1]))

    runs = load_runs(a.run)
    ours = Ours(ans)
    xs = np.linspace(-1.2, 1.2, 3001)
    slow = ours_at(ans, xs)
    fast = ours(xs)
    print("  fast evaluation against ray-by-ray integration, 3001 rays: max |diff| %.1e"
          % max(np.max(np.abs(fast[v] - slow[v])) for v in VARS))
    jumps = (sp[1], sp[2], sp[3])
    p0, p1 = P[1], P[2]
    print("\nL1 against our answer (in brackets: more than 8 cells from every discontinuity)"
          "\nand, behind the left slow wave, the mean of (q - plateau) / |plateau - ahead|"
          " in xi 0.033-0.037 (inside the compound fan) and 0.040-0.050:")
    for key in sorted(runs):
        r = runs[key]
        ex = ours(r["xi"])
        dx = T * (r["xi"][1] - r["xi"][0])
        away = np.ones(r["xi"].size, bool)
        for s in jumps:
            away &= np.abs(r["xi"] - s) * T > 8 * dx
        l1 = "  ".join("%s %.2e (%.2e)" % (v, np.abs(r[v] - ex[v]).mean(),
                                            np.abs(r[v] - ex[v])[away].mean()) for v in ("rho", "By"))
        win = []
        for lo, hi in ((0.033, 0.037), (0.040, 0.050)):
            m = (r["xi"] > lo) & (r["xi"] < hi)
            win.append(" ".join("%s %+.3f" % (v, np.mean(r[v][m] - p1[v]) / abs(p1[v] - p0[v]))
                                for v in ("rho", "By")))
        print("  %-4s %-6s %6d   %s   | %s | %s" % (key + (l1, win[0], win[1])))


# ── figures ──────────────────────────────────────────────────────────────────

COLORS = ["#F2B279", "#D9822B", "#8A4510", "#4F86C6", "#2E5E99", "#16213A"]


def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ans = answers(a.answers)
    sp = ans["sp"]
    runs = load_runs(a.run)
    sampler = PR.make_sampler(G)
    x = np.linspace(0.0, 1.0, 8001)
    ex = ours_at(ans, (x - 0.5) / T, sampler)
    zoom = [run_key(s) for s in a.zoom]
    style = {}
    for i, k in enumerate(zoom):
        style[k] = dict(color="#8C8C8C" if k[0] == "hlle" else COLORS[i % len(COLORS)],
                        label="%s/%s %d cells" % (k[0].upper(), k[1].upper(), k[2]))

    # 1. the profiles, with zooms on the left slow wave
    full = runs[run_key(a.full)]
    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 4)
    for k, (key, lab) in enumerate((("rho", "ρ"), ("p", "p (gas)"), ("vx", "v_x"),
                                    ("vy", "v_y"), ("By", "B_y"))):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        ax.plot(full["x"], full[key], "o", ms=2.2, mfc="none", mec="#D9822B", mew=0.7,
                label="%s, %s, %d cells" % (run_key(a.full)[0].upper(), run_key(a.full)[1].upper(),
                                             run_key(a.full)[2]))
        ax.plot(x, ex[key], "-", color="k", lw=1.1, label="exact solution (our solver)")
        ax.set_title(lab, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        if key in ("By", "rho"):
            ax.axvspan(0.505, 0.522, color="#E8D5F0", alpha=0.6, lw=0)
        if k == 0:
            ax.legend(fontsize=8.5, loc="lower left")
        if k == 4:
            ax.set_xlabel("x")
    mz = (x >= 0.505) & (x <= 0.522)
    for col, key, title in ((1, "By", "B_y at the left slow wave (zoom)"),
                            (2, "rho", "ρ at the left slow wave (zoom)")):
        ax = fig.add_subplot(gs[1, col])
        for k in zoom:
            r = runs[k]
            m = (r["x"] > 0.505) & (r["x"] < 0.522)
            ax.plot(r["x"][m], r[key][m], ".-", ms=3.2, lw=0.8, **style[k])
        ax.plot(x[mz], ex[key][mz], "-", color="k", lw=1.6, label="exact solution (our solver)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.505, 0.522)
        ax.set_xticks([0.506, 0.510, 0.514, 0.518, 0.522])
        ax.set_xlabel("x")
        ax.set_title(title, fontsize=12)
        if key == "By":
            ax.legend(fontsize=7.6, loc="center right")
    tx = fig.add_subplot(gs[1, 3])
    tx.axis("off")
    tx.text(0.0, 0.98,
            "Balsara 1 (relativistic Brio–Wu), Γ = 2, t = 0.4\n"
            "L: ρ = 1, p = 1, B = (0.5, 1, 0);  R: ρ = 0.125, p = 0.1, B = (0.5, −1, 0)\n\n"
            "Waves, left to right: fast rarefaction, a slow-family\n"
            "shock that REVERSES B_y (+0.66 → −0.43), contact, slow\n"
            "shock, weak fast rarefaction. Both Alfvén waves carry\n"
            "nothing.\n\n"
            "The reversing wave is an intermediate shock: the gas\n"
            "crosses the Alfvén speed through it. Checked: jump\n"
            "conditions to 1e-13, entropy rises across both shocks,\n"
            "the runs converge to it, and it is the only root found\n"
            "from 224 starts (none with a 180° Alfvén rotation).\n\n"
            "The spike in ρ, p behind it is the scheme's overshoot,\n"
            "about two cells wide at every resolution.",
            transform=tx.transAxes, va="top", fontsize=9.6)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=140)
    fig.savefig(a.out + ".pdf")
    plt.close(fig)

    # 2. the left slow wave on its own: our shock against the compound answer
    xif = np.linspace(ans["Vsh"], ans["Vtl"], 600)
    fan, chk = compound_fan(ans, xif, sampler)
    print("compound fan: %d rays failed; |slow speed - ray| max %.1e; sonic head %.1e; "
          "tail vs contact state %.1e" % (chk["failed"], chk["ray_vs_slow_speed"],
                                          chk["sonic_head"], chk["tail_vs_B"]))
    pA, pS, pB = prim(ans["A"]), prim(ans["S"]), prim(ans["B"])
    p0, p1 = prim(ans["Z"][0]), prim(ans["Z"][1])
    XS = 0.5 + T * sp[1]
    X0, X1 = 0.5105, 0.5175
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True)
    for ax, (key, lab) in zip(axs.ravel(), (("By", "B_y"), ("rho", "ρ"), ("p", "p (gas)"),
                                            ("vy", "v_y"))):
        ax.axvspan(0.5 + T * ans["Vsh"], 0.5 + T * ans["Vtl"], color="#E8D5F0", alpha=0.7, lw=0)
        for k in zoom:
            r = runs[k]
            m = (r["x"] > X0) & (r["x"] < X1)
            ax.plot(r["x"][m], r[key][m], "s" if k[0] == "hlle" else "o",
                    ms=2.6 if k[0] == "hlle" else 3.2, color=style[k]["color"], label=style[k]["label"])
        ax.plot([X0, XS, XS, X1], [p0[key], p0[key], p1[key], p1[key]], "-", color="k", lw=1.8,
                label="exact: single intermediate shock (our solver)")
        cx = [X0, 0.5 + T * ans["Vsh"], 0.5 + T * ans["Vsh"]] + list(0.5 + T * xif) + [X1]
        cy = [pA[key], pA[key], pS[key]] + list(np.asarray(fan[key])) + [pB[key]]
        ax.plot(cx, cy, "-", color="#7B3294", lw=2.0, label="exact: compound (shock + attached slow fan)")
        ax.set_title(lab, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(X0, X1)
    for ax in axs[1]:
        ax.set_xlabel("x  (t = 0.4)")
    axs[0, 0].legend(fontsize=7.8, loc="upper right")
    fig.suptitle("Balsara 1, the left slow wave: our single intermediate shock (black) and the compound\n"
                 "alternative (purple: a sonic intermediate shock, then a slow rarefaction, shaded, "
                 "%.4f wide)" % (T * (ans["Vtl"] - ans["Vsh"])), fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out + "_fan.png", dpi=140)
    fig.savefig(a.out + "_fan.pdf")
    plt.close(fig)
    print("wrote %s.png, %s_fan.png (and .pdf)" % (a.out, a.out))


# ── roots ────────────────────────────────────────────────────────────────────

def roots(a):
    from rmhd.batched import planar5_b as P5
    flips = {"00": (False, False), "10": (True, False), "01": (False, True), "11": (True, True)}
    S5 = P5.make_solver(G)
    pl = np.linspace(np.log(0.5), np.log(0.9), 4)
    bt = np.linspace(-0.95, 0.95, 14)
    seeds = np.array(np.meshgrid(pl, bt, pl, indexing="ij")).reshape(3, -1)
    K = seeds.shape[1]
    L = [np.full(K, float(c)) for c in LEFT]
    R = [np.full(K, float(c)) for c in RIGHT]
    Bn = np.full(K, float(Bx))
    for name in a.flips:
        fl = flips[name]
        r = S5["solve"](L, R, Bn, unk3_init=[c.copy() for c in seeds], accuracy=1e-10,
                        verify_tol=1e-8, flip=fl)
        conv = np.asarray(r["converged"], dtype=bool)
        print("flip %s (pi-rotation at the left / right Alfven wave): %d of %d starts converge"
              % (fl, int(conv.sum()), K), flush=True)
        if not conv.any():
            continue
        U = np.stack([np.asarray(c) for c in r["unk"]], 1)[conv]
        lanes = np.flatnonzero(conv)
        uniq, idx, cnt = np.unique(np.round(U, 6), axis=0, return_index=True, return_counts=True)
        for j, c in zip(idx, cnt):
            u = U[j]
            print("  root from %d starts: p_LF %.9f  Bt* %+.9f  p_RF %.9f  speeds %s"
                  % (c, np.exp(u[0]), u[1], np.exp(u[2]),
                     np.round([float(np.asarray(v)[lanes[j]]) for v in r["speeds"]], 5)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("checks", "figures"):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", default=[], metavar="FLUX:FILE")
        p.add_argument("--answers", default=None,
                       help="file with the exact answers (default: the first --run file)")
        if name == "figures":
            p.add_argument("--full", required=True, metavar="FLUX:LIMITER:N")
            p.add_argument("--zoom", nargs="+", required=True, metavar="FLUX:LIMITER:N")
            p.add_argument("--out", required=True, help="output stem")
    p = sub.add_parser("roots")
    p.add_argument("flips", nargs="+", choices=("00", "10", "01", "11"))
    a = ap.parse_args()
    if getattr(a, "answers", None) is None and a.cmd != "roots":
        a.answers = a.run[0].split(":", 1)[1]
    {"checks": checks, "figures": figures, "roots": roots}[a.cmd](a)


if __name__ == "__main__":
    main()
