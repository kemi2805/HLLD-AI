#!/usr/bin/env python
"""Two exact answers of one face (ledger lane 1013): the four-unknown answer
whose slow jump runs across the Alfven speed (what the gate refuses), and the
regular seven-wave answer with the rotation in front of the slow wave.  B_t and
the density against xi = x/t around the left and the right slow/Alfven pair,
with the face's own 16384-cell tube.  -> figs/two_answers_1013.png

    PYTHONPATH=rmhd_fix python figs_two_answers.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import refused_data as CR
import sol_profiles as SP
import verify_rmhd as VR
import rmhd_ideal as RI

LANE = 1013
g, Bn = CR.g, float(CR.pop["UL"][LANE][7])
NAMES = VR.NAMES
SW = {0: "LF", 2: "LS", 4: "RS", 6: "RF"}


def answer_profile(Z, V, kind):
    """(xi, states) of a seven-wave answer: plateaus, jumps, fans sampled on
    their own integral curves."""
    B1 = np.array([Bn])
    pieces = []
    edges = []
    for w in range(7):
        left = w < 3
        ia, ib = (w, w + 1) if left else (w + 1, w)
        if kind[w] == "fan" and w in SW:
            A = [np.array([Z[ia, c]]) for c in range(7)]
            B = [np.array([Z[ib, c]]) for c in range(7)]
            y, xi = SP.fan_states(A, B, SW[w], 2 if w in (2, 4) else 1, B1, k=400)
            o = np.argsort(xi)
            pieces.append((xi[o], np.stack([c[o] for c in y])))
            edges.append((xi.min(), xi.max()))
        else:
            edges.append((V[w], V[w]))
    lo = -1.0
    for j in range(8):
        hi = edges[j][0] if j < 7 else 1.0
        pieces.append((np.array([lo, hi]), np.repeat(Z[j][:, None], 2, axis=1)))
        if j < 7:
            lo = edges[j][1]
    pieces.sort(key=lambda p: p[0][0])
    xi = np.concatenate([p[0] for p in pieces])
    st = np.concatenate([p[1] for p in pieces], axis=1)
    return xi, st


def main():
    Zr, Sr = CR.ans[LANE]                                   # the refused answer
    UL8 = np.concatenate([Zr[0], [Bn]])[None]
    UR8 = np.concatenate([Zr[7], [Bn]])[None]
    kr = VR.verify(UL8, UR8, Zr[None], Sr[None], g)["kind"][0]
    rows = [r for r in np.load(CR.D + "/rd_solve_test.npy", allow_pickle=True)
            if r["lane"] == LANE and r["adm"]]
    reg = rows[0]
    xr, sr = answer_profile(Zr, Sr, kr)
    xg, sg = answer_profile(reg["Z"], reg["V"], reg["kind"])
    z = np.load(CR.D + "/tubes_hi/tube_%d_16384.npz" % LANE)
    t, nc = float(z["t"]), 16384
    xt = ((np.arange(nc) + 0.5) / nc - 0.5) / t

    # the face flux of each (the state at xi = 0 is the plateau behind LF)
    F = lambda X: RI.cons_flux([np.array([c]) for c in X], np.array([Bn]), g)[1][:, 0]
    i0r, i0g = np.searchsorted(xr, 0.0), np.searchsorted(xg, 0.0)
    dF = np.max(np.abs(F(sg[:, i0g]) - F(sr[:, i0r]))) / np.max(np.abs(F(sr[:, i0r])))

    C_REF, C_REG, C_TUBE = "#c2410c", "#1d4ed8", "#6b7280"
    fig, ax = plt.subplots(2, 3, figsize=(14, 6.8), constrained_layout=True)
    panels = (("left pair, wide: the tube", 1, 2, 6e-4, 1e-4, True),
              ("left pair, close up", 1, 2, None, 1e-6, False),
              ("right pair, close up", 5, 4, None, 1e-6, False))
    for col, (title, wrot, wslow, half, unit, with_tube) in enumerate(panels):
        Va = reg["V"][wrot]
        if half is None:
            # the sliver: the refused jump, the rotation and the whole slow fan
            ib = 3 if wslow == 2 else 4                # the state behind the slow fan
            tail = float(SP.XI([np.array([reg["Z"][ib, c]]) for c in range(7)],
                               SW[wslow], np.array([Bn]))[0])
            span = max(abs(Sr[wslow] - Va), abs(tail - Va), 2e-6)
            half = 1.15 * span
        for row, (comp, name) in enumerate(((5, "tangential field $B_t$"), (0, "density"))):
            a = ax[row, col]
            sc = lambda x: (x - Va) / unit
            if with_tube:
                m = (xt > Va - half) & (xt < Va + half)
                a.plot(sc(xt[m]), z["By" if comp == 5 else "rho"][m], "o", ms=6, color=C_TUBE,
                       label="tube, 16384 cells (one cell = %.1e)" % (1.0 / (nc * t)))
            a.plot(sc(xr), sr[comp], "-", lw=2.6, color=C_REF, label="refused: one slow jump across the Alfven speed")
            a.plot(sc(xg), sg[comp], "--", lw=2.0, color=C_REG, label="regular: rotation + slow fan")
            a.set_xlim(-half / unit, half / unit)
            w = (np.abs(xr - Va) < half)
            wg = (np.abs(xg - Va) < half)
            lo = min(sr[comp][w].min(), sg[comp][wg].min()); hi = max(sr[comp][w].max(), sg[comp][wg].max())
            pad = 0.08 * (hi - lo + 1e-300)
            a.set_ylim(lo - pad, hi + pad)
            a.set_xlabel("$x/t$ minus the Alfven speed (units of %.0e)" % unit)
            a.set_ylabel(name)
            a.grid(alpha=0.25, lw=0.6)
            if row == 0:
                a.set_title(title, fontsize=10)
            if row == 0 and col in (0, 1):
                a.legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle("Lane %d: two exact answers of one face. They differ only inside a sliver a few 1e-5 wide around "
                 "each slow/Alfven pair (a tube cell is 3.4e-4); the face flux differs by %.0e" % (LANE, dF), fontsize=11)
    out = CR.D + "/figs/two_answers_%d.png" % LANE
    fig.savefig(out, dpi=150)
    print(out, "face flux difference %.1e" % dF)
    for w in range(7):
        print("  %s refused %s V %.9f | regular %s V %.9f" % (NAMES[w], kr[w], Sr[w], reg["kind"][w], reg["V"][w]))
    # plateau differences
    for j, nm in enumerate(("L", "R2", "R3", "R4", "R5", "R6", "R7", "R")):
        d = np.max(np.abs(reg["Z"][j] - Zr[j]) / np.maximum(np.abs(Zr[j]), 1e-3))
        print("  zone %-2s max rel difference %.1e" % (nm, d))


if __name__ == "__main__":
    main()
