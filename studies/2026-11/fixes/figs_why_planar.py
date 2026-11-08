#!/usr/bin/env python
"""Figures for the user (2026-11-05, "As long as the wave speeds are not the
same, the code should be able to calculate the 7 wave structure" -- "please
plot me some solutions like this"):

    figs/why_planar_solutions.png  four 4Q faces, B_t(x/t) of the exact
        solution with all seven waves marked, and a zoom on the closest
        Alfven / slow pair:
          a) solved by the six-unknown Newton: a W = 7 shear whose Alfven and
             slow speeds are 4e-5 apart -- crowding alone does not stop it;
          b) answered by the five-wave planar solver: nearly at rest, every
             speed 0.25 c from its neighbour -- the Newton never started from
             production's seed, and converges from the 4Q network's;
          c) ledger lane 1013 (lost by production): Alfven and slow speeds
             1e-7 / 1e-8 apart, a pi turn in front of each slow fan;
          d) ledger lane 215 (lost by production): B_t through zero inside
             both slow fans.
    figs/why_planar_gaps.png  the Alfven-slow gap of the outer states, by
        who answers the face (4Q 64^2 control, first three harvest shards)
        and for the faces production lost (4Q 128^2 failure ledger).

    PYTHONPATH=rmhd_fix python figs_why_planar.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
sys.path.insert(0, F)
import sol_profiles as SP                 # noqa: E402  (ledger population, gamma 5/3)
import verify_rmhd as VR                  # noqa: E402
from rmhd.batched import wave_speeds_b as WB   # noqa: E402

np.seterr(all="ignore")
OUT = F + "/figs"
G = SP.GAMMA
LEDGER = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz",
                 allow_pickle=True)
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"
EXACT, FAN, ROT = "#111827", "#eda100", "#e87ba4"
C_NEWTON, C_P5, C_LOST = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3, validated
NAMES = ("LF", "LA", "LS", "C", "RS", "RA", "RF")
SW = {0: "LF", 2: "LS", 4: "RS", 6: "RF"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


# ── the four answers, each as (Z[8, 7] lab states, V[7] speeds, B_n) ────────
def harvest_face(shard, i):
    a = np.load(F + "/diag_4q/solved_r000_%05d.npz" % shard, allow_pickle=True)
    Z, V = np.array(a["zones"][i], float), np.array(a["speeds"][i], float)
    assert np.allclose(Z[0], a["U_L"][i][:7]) and np.allclose(Z[7], a["U_R"][i][:7])
    return Z, V, float(a["U_L"][i][7])


def resolve_face(lane):
    """The regular admissible answer the rotation re-solve found (rd_solve)."""
    rows = np.load(F + "/rd_solve_test.npy", allow_pickle=True)
    r = [r for r in rows if r["lane"] == lane and r["adm"]][0]
    return np.array(r["Z"], float), np.array(r["V"], float), float(LEDGER["UL"][lane][7])


def alfven(X, Bn):
    _, la, ra, _ = WB.xi_all(*[np.array([c]) for c in X], np.array([Bn]), G)
    return float(np.asarray(la).ravel()[0]), float(np.asarray(ra).ravel()[0])


def planar4_face(lane, kinds, pl, pr):
    """planar4's answer (no rotations: A2 = A, D2 = D) in the seven-wave layout."""
    sol = SP.solve(lane, kinds, pl, pr)
    al, Bn = sol["alpha"], float(sol["Bn"][0])
    lab = lambda X: np.array([c[0] for c in SP.lab(X, al)])
    A, B, C, D = sol["zones"]
    Z = np.stack([lab(sol["L0"]), lab(A), lab(A), lab(B), lab(C), lab(D), lab(D), lab(sol["R0"])])
    V = np.array([sol["V"][0], alfven(Z[1], Bn)[0], sol["V"][1], sol["V"][2],
                  sol["V"][3], alfven(Z[6], Bn)[1], sol["V"][4]])
    return Z, V, Bn


def kinds_of(Z, V, Bn):
    UL8 = np.concatenate([Z[0], [Bn]])[None]
    UR8 = np.concatenate([Z[7], [Bn]])[None]
    return VR.verify(UL8, UR8, Z[None], V[None], G)["kind"][0]


def profile(Z, V, kind, Bn):
    """(xi, states[7, n], edges): plateaus, jumps, fans sampled along their own
    integral curves (slow fans in the signed field, else in the pressure)."""
    B1 = np.array([Bn])
    pieces, edges = [], []
    for w in range(7):
        ia, ib = (w, w + 1) if w < 3 else (w + 1, w)
        if kind[w] == "fan" and w in SW:
            A = [np.array([Z[ia, c]]) for c in range(7)]
            B = [np.array([Z[ib, c]]) for c in range(7)]
            for kk in ((2, 1) if w in (2, 4) else (1,)):
                y, xi = SP.fan_states(A, B, SW[w], kk, B1, k=600)
                end = np.array([c[-1] for c in y])
                if np.all(np.isfinite(end)) and np.max(np.abs(end - Z[ib])) < 1e-6 * (1 + np.abs(Z[ib]).max()):
                    break
            o = np.argsort(xi)
            pieces.append((xi[o], np.stack([c[o] for c in y])))
            edges.append((float(xi.min()), float(xi.max())))
        else:
            edges.append((float(V[w]), float(V[w])))
    lo = -1.0
    for j in range(8):
        hi = edges[j][0] if j < 7 else 1.0
        pieces.append((np.array([lo, hi]), np.repeat(Z[j][:, None], 2, axis=1)))
        if j < 7:
            lo = edges[j][1]
    pieces.sort(key=lambda p: p[0][0])
    return (np.concatenate([p[0] for p in pieces]),
            np.concatenate([p[1] for p in pieces], axis=1), edges)


def beta(X, Bn):
    rho, Pt, vx, vy, vz, By, Bz = X
    W2 = 1.0 / (1.0 - vx * vx - vy * vy - vz * vz)
    b2 = (Bn * Bn + By * By + Bz * Bz) / W2 + (vx * Bn + vy * By + vz * Bz) ** 2
    return 2.0 * (Pt - 0.5 * b2) / b2


def lorentz(X):
    return 1.0 / np.sqrt(1.0 - X[2] ** 2 - X[3] ** 2 - X[4] ** 2)


def gap_between(edges, a, b):
    """Distance between waves a and b (nearest edges, fans included)."""
    (a0, a1), (b0, b1) = edges[a], edges[b]
    return max(b0 - a1, a0 - b1, 0.0)


# ── panels ──────────────────────────────────────────────────────────────
def mark_waves(ax, edges, kind, lo, hi, scale=1.0, off=0.0):
    """A thin line (fans: a band) at each wave inside [lo, hi] and its name
    above the axes, staggered where names would collide."""
    placed = []
    for w in range(7):
        a, b = edges[w]
        if b < lo or a > hi:
            continue
        rot = w in (1, 5)
        zero = kind[w] == "zero"
        xa, xb = (a - off) / scale, (b - off) / scale
        if b > a:
            ax.axvspan(xa, xb, color=FAN, alpha=0.28, lw=0)
        else:
            ax.axvline(xa, color=ROT if (rot and not zero) else MUTED,
                       lw=1.6 if (rot and not zero) else 0.7, ls=":" if zero else "-",
                       alpha=0.9 if not zero else 0.6, zorder=1)
        xm = 0.5 * (xa + xb)
        frac = (xm - (lo - off) / scale) / ((hi - lo) / scale)
        lvl = 0
        while any(abs(frac - f) < 0.045 and l == lvl for f, l in placed):
            lvl += 1
        placed.append((frac, lvl))
        name = NAMES[w] + (" turn" if (rot and not zero) else "")
        ax.text(xm, 1.02 + 0.075 * lvl, name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8.2, color=INK if not zero else MUTED,
                fontweight="bold" if (rot and not zero) else "normal", clip_on=False)


def unit_label(x0, scale):
    sgn = "+" if x0 < 0 else "-"
    return "(x/t %s %.6f) in units of %.0e" % (sgn, abs(x0), scale)


def window_plot(ax, xi, y, lo, hi, x0, scale, edges, kind, names=True):
    """y(xi) inside [lo, hi] (plateaus carried to both ends), in units of
    ``scale`` relative to ``x0``, with the waves marked."""
    m = (xi >= lo) & (xi <= hi)
    xs = np.concatenate([[lo], xi[m], [hi]])
    ys = np.concatenate([[np.interp(lo, xi, y)], y[m], [np.interp(hi, xi, y)]])
    ax.plot((xs - x0) / scale, ys, color=EXACT, lw=1.8)
    if ys.min() < 0 < ys.max():
        ax.axhline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_xlim((lo - x0) / scale, (hi - x0) / scale)
    pad = 0.12 * (ys.max() - ys.min() + 1e-12)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    if names:
        mark_waves(ax, edges, kind, lo, hi, scale=scale, off=x0)
    else:
        for w in range(7):
            a, b = edges[w]
            if b < lo or a > hi:
                continue
            if b > a:
                ax.axvspan((a - x0) / scale, (b - x0) / scale, color=FAN, alpha=0.28, lw=0)
            else:
                rot = w in (1, 5) and kind[w] != "zero"
                ax.axvline((a - x0) / scale, color=ROT if rot else MUTED, lw=1.4 if rot else 0.8,
                           ls=":" if kind[w] == "zero" else "-")


def draw_face(sf, Z, V, Bn, title, side, right, inset_loc):
    """One face: B_t over the whole fan (left) and, on the right, either a zoom
    on one side's Alfven / slow pair (with an inset at the scale of their gap
    when the zoom cannot show it) or the density over the whole fan."""
    kind = kinds_of(Z, V, Bn)
    xi, st, edges = profile(Z, V, kind, Bn)
    Bt = st[5]
    axf, axr = sf.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    sf.suptitle(title, x=0.005, ha="left", fontsize=10, color=INK)
    lo, hi = max(edges[0][0] - 0.06, -1), min(edges[6][1] + 0.06, 1)
    window_plot(axf, xi, Bt, lo, hi, 0.0, 1.0, edges, kind)
    axf.set_ylabel("B_t (lab)")
    axf.set_xlabel("x / t")
    if right == "density":
        window_plot(axr, xi, st[0], lo, hi, 0.0, 1.0, edges, kind)
        axr.set_ylabel("density (lab)")
        axr.set_xlabel("x / t")
        axr.set_title("density: an ordinary seven-wave structure -- nothing close enough to zoom into",
                      loc="left", fontsize=9.2, color=MUTED, pad=24)
        return kind, edges
    wa, ws = (1, 2) if side == "L" else (5, 4)
    a, b = sorted((wa, ws))
    g0, g1 = edges[a][1], edges[b][0]                   # the gap between the pair
    gap = max(g1 - g0, 0.0)
    span = max(edges[wa][1], edges[ws][1]) - min(edges[wa][0], edges[ws][0])
    pad = max(0.6 * span, 3 * gap, 1e-12)
    zlo = min(edges[wa][0], edges[ws][0]) - pad
    zhi = max(edges[wa][1], edges[ws][1]) + pad
    scale = 10.0 ** np.floor(np.log10(zhi - zlo))
    x0 = edges[wa][0]
    window_plot(axr, xi, Bt, zlo, zhi, x0, scale, edges, kind)
    axr.set_xlabel(unit_label(x0, scale))
    axr.set_title("zoom: the %s Alfven and slow waves, %.1e apart" % ("left" if side == "L" else "right", gap),
                  loc="left", fontsize=9.2, color=MUTED, pad=24)
    if gap < 0.05 * (zhi - zlo):
        # the gap itself, at its own scale
        ilo, ihi = g0 - 2.0 * gap, g1 + 2.0 * gap
        iscale = 10.0 ** np.floor(np.log10(ihi - ilo))
        ins = axr.inset_axes(inset_loc)
        window_plot(ins, xi, Bt, ilo, ihi, x0, iscale, edges, kind, names=False)
        ins.set_title("the gap, in units of %.0e" % iscale, fontsize=8, color=MUTED, pad=3)
        ins.tick_params(labelsize=7.5)
        ins.grid(True, color=GRID, lw=0.5)
        for sp in ins.spines.values():
            sp.set_edgecolor(MUTED)
    return kind, edges


def solutions():
    rp = np.load(F + "/diag_4q/p5_0.npz")
    re_ = np.load(F + "/diag_4q/early_0.npz")
    off = np.load(F + "/replay_final2/replay_off.npz")
    ps = lambda l: float(abs(off["pstar"][l]))

    faces = []
    Z, V, Bn = harvest_face(0, 0)
    faces.append((Z, V, Bn, "a) Solved by the six-unknown Newton at its first try (4Q 64^2, first steps): a shear with "
                  "W = %.1f on the left. Its slow fan starts %.0e after the Alfven speed --\n"
                  "    crowded speeds alone do not stop the Newton" % (lorentz(Z[0]), abs(V[2] - V[1])),
                  "L", "zoom", [0.45, 0.1, 0.5, 0.45]))
    Z, V, Bn = harvest_face(0, 611)
    g_all = min(abs(V[k + 1] - V[k]) for k in range(6))
    faces.append((Z, V, Bn, "b) Answered by the five-wave planar solver (4Q 64^2, first steps): nearly at rest "
                  "(W = %.2f), every wave speed at least %.2f c from its neighbour.\n"
                  "    The Newton never took a step from production's seed (%d iterations); from the 4Q "
                  "network's seed it converges in %d" % (
                      max(lorentz(Z[0]), lorentz(Z[7])), g_all, rp["iters7"][0], re_["iters7"][56]),
                  "R", "density", None))
    Z, V, Bn = resolve_face(1013)
    faces.append((Z, V, Bn, "c) Ledger face 1013, lost by production (4Q 128^2): beta = %.0f, B_t / B_n = %.2f; Alfven "
                  "and slow speeds %.0e (left) / %.0e (right) apart. Each slow fan has a pi turn\n"
                  "    in front of it; on the right the turn keeps the lab sign of B_t (%+.3f -> %+.3f) and the fan "
                  "behind it carries B_t through zero" % (
                      beta(Z[0], Bn), abs(Z[0][5] / Bn), abs(V[2] - V[1]), abs(V[5] - V[4]), Z[6][5], Z[5][5]),
                  "R", "zoom", [0.06, 0.5, 0.45, 0.42]))
    Z, V, Bn = planar4_face(215, (1, 1), ps(215), ps(215))
    faces.append((Z, V, Bn, "d) Ledger face 215, lost by production (4Q 128^2): beta = %.0f, B_t / B_n = %.2f. "
                  "B_t passes through zero inside both slow fans --\n"
                  "    the six-unknown form, which steers each slow wave by the magnitude |B_t|, cannot follow it" % (
                      beta(Z[0], Bn), abs(Z[0][5] / Bn)),
                  "L", "zoom", None))

    fig = plt.figure(figsize=(15.5, 4.7 * len(faces)), layout="constrained")
    subs = fig.subfigures(len(faces), 1, hspace=0.035)
    for sf, (Z, V, Bn, title, side, right, iloc) in zip(subs, faces):
        kind, edges = draw_face(sf, Z, V, Bn, title, side, right, iloc)
    fig.suptitle("Seven-wave solutions of four 4Q faces.  Shaded = rarefaction fans;  pink = an Alfven wave that "
                 "turns the field by pi;  dotted = an Alfven wave of zero strength.\n"
                 "Every answer is a full seven-wave solution of ideal RMHD; a) and b) are cheap for the right "
                 "start, c) and d) need the planar form.", fontsize=11.5, color=INK, x=0.005, ha="left")
    fig.savefig(os.path.join(OUT, "why_planar_solutions.png"), dpi=130)
    plt.close(fig)


def outer_gap(U):
    """min over the two sides of |lambda_Alfven - lambda_slow| of the outer state
    (left-going on the left, right-going on the right)."""
    def side(X, s):
        eig, la, ra, _ = WB.xi_all(*[X[:, j] for j in range(7)], X[:, 7], G)
        eig = np.asarray(eig)
        eig = eig if eig.shape[0] == X.shape[0] else eig.T
        return np.abs(la - eig[:, 1]) if s == "L" else np.abs(ra - eig[:, 2])
    UL, UR = U
    return np.minimum(side(UL, "L"), side(UR, "R"))


def gaps():
    newton, p5 = [], []
    for sh in (0, 1, 2):
        a = np.load(F + "/diag_4q/solved_r000_%05d.npz" % sh, allow_pickle=True)
        src = np.asarray(a["source"]).astype(int)
        gp = outer_gap((a["U_L"], a["U_R"]))
        newton.append(gp[src == 0])
        p5.append(gp[src == 1])
    newton, p5 = np.concatenate(newton), np.concatenate(p5)
    m = LEDGER["group"] == 1
    lost = outer_gap((LEDGER["UL"][m], LEDGER["UR"][m]))
    series = (("solved by the six-unknown Newton (4Q 64^2, early)", newton, C_NEWTON),
              ("answered by the five-wave planar solver (same)", p5, C_P5),
              ("lost by production (4Q 128^2 ledger)", lost, C_LOST))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axvline(1e-4, color=MUTED, lw=0.8, ls="--")
    rows = []
    for name, gp, col in series:
        x = np.sort(np.clip(gp, 1e-12, None))
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", color=col, lw=2.0, label="%s, %s faces" % (name, format(x.size, ",")))
        rows.append((name, np.mean(gp < 1e-4), np.median(gp)))
    ax.set_xscale("log")
    ax.set_xlim(1e-10, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("gap between the Alfven and the slow speed of the outer states (the smaller side), in c")
    ax.set_ylabel("fraction of faces with a smaller gap")
    ax.legend(loc="upper left", fontsize=9)
    txt = "\n".join("%s: median %.0e, %.0f%% below 1e-4" % (n.split(" (")[0], md, 100 * f) for n, f, md in rows)
    ax.text(0.985, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
    ax.text(1.15e-4, 0.015, "1e-4", color=MUTED, fontsize=8.5, va="bottom")
    ax.set_title("How close the Alfven and slow speeds are, by who answers the face", loc="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "why_planar_gaps.png"), dpi=140)
    plt.close(fig)
    for n, f, md in rows:
        print("%-60s median %.1e  below 1e-4 %.1f%%" % (n, md, 100 * f))


if __name__ == "__main__":
    solutions()
    gaps()
