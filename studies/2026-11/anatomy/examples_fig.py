#!/usr/bin/env python
"""Visual examples of faces the four-quadrant 128^2 run lost (2026-11-04; user:
"please give me visual examples of the loss").

For each chosen lost face (failure ledger 78407 lanes, located on the grid
through face_table.py): where it sits (log rho of the nearest snapshot, the
face and the other lost faces of that moment marked), and its own Riemann
problem evolved as a 1D tube (src/physics/tube_seed.run_tubes: HLLD, MC,
4096 and 1024 cells) -- the structure the PDE builds where no exact answer
exists -- against x/t, with the characteristic speeds of the two states.
The flux the run used there (HLLD on the two states) is compared with the
tube's at x/t = 0.

    HLLD_DIR=... python examples_fig.py LANE [LANE ...] --out figs/lost_examples.png
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HLLD = os.environ.get("HLLD_DIR", "/Users/miler/Codes/HLLD")
sys.path.insert(0, HLLD)
sys.path.insert(0, HERE)

INK, MUTED = "#222222", "#666666"
SPEED = {"fast": "#7f7f7f", "Alfvén": "#CC79A7", "slow": "#0072B2"}
RUN = os.path.join(os.path.dirname(HERE), "runs64", "ft3_128")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lanes", type=int, nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--ncells", type=int, nargs="+", default=[4096, 2048, 1024])
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "lost_examples.png"))
    a = ap.parse_args()
    import torch
    from rmhd.batched import wave_speeds_b as WB
    from src.physics.eos import hybrid_eos
    from src.physics.hlld import hlld_flux, compute_srmhd_fluxes
    from src.physics.tube_seed import run_tubes
    from feats import load
    p = np.load(os.path.join(HERE, "data", "ledger", "population.npz"), allow_pickle=True)
    loc = np.load(os.path.join(HERE, "data", "ledger", "lanes_located.npz"))
    lt = np.load(os.path.join(HERE, "data", "ledger", "lanes_typed.npz"))
    gamma = float(p["gamma"])
    eos = hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)
    d = load(os.path.join(HERE, "data", "sample.npz"))
    lost = d["group"] == 4
    snaps = []
    for f in sorted(glob.glob(os.path.join(RUN, "snap_0*.npz"))):
        z = np.load(f, allow_pickle=True)
        snaps.append({k: z[k] for k in ("t", "x", "y", "rho")})
    lanes = np.array(a.lanes)
    UL, UR = p["UL"][lanes], p["UR"][lanes]
    Bn = UL[:, 7]
    prof = {}                                       # ncells -> (profiles per lane, t)
    for nc in a.ncells:
        cols, ts = {}, []
        for ln in lanes:
            z = np.load(os.path.join(HERE, "data", "tubes", "tube_%d_%d.npz" % (ln, nc)))
            for k_ in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
                cols.setdefault(k_, []).append(z[k_])
            ts.append(float(z["t"]))
        prof[nc] = ({k_: np.stack(v, axis=1) for k_, v in cols.items()}, np.array(ts))
    # where each lane sits: its states matched to the face table
    keyf = lambda U1, U2: np.ascontiguousarray(np.concatenate([U1, U2], axis=1)).view(np.dtype((np.void, 128))).ravel()
    kf = keyf(d["U_L"], d["U_R"])
    where = {}
    for i_, k_ in enumerate(kf):
        where.setdefault(k_.tobytes(), i_)
    fidx = [where.get(k_.tobytes(), -1) for k_ in keyf(UL, UR)]
    eig, la, ra, ok = WB.xi_all(*[UL[:, j] for j in range(7)], Bn, gamma)
    eigR, laR, raR, okR = WB.xi_all(*[UR[:, j] for j in range(7)], Bn, gamma)
    eig, eigR = np.asarray(eig), np.asarray(eigR)
    n = lanes.size
    fig, ax = plt.subplots(n, 5, figsize=(22, 3.75 * n), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1.0, 1.2, 1.2, 1.2, 1.2]))
    ax = np.atleast_2d(ax)
    nc0 = a.ncells[0]
    for r, ln in enumerate(lanes):
        fi = fidx[r]
        t0, x0, y0 = float(d["t"][fi]), float(d["x"][fi]), float(d["y"][fi])
        typ = lt["typ"][ln]
        ver = lt["verdict"][ln] if lt["fail"][ln] else "solved (%s)" % {0: "seven-wave", 1: "seven-wave after retry",
                                                                     2: "planar rescue", 3: "four-unknown"}[int(d["group"][fi])]
        # (1) where
        s = min(snaps, key=lambda s_: abs(float(s_["t"]) - t0))
        A = ax[r, 0]
        A.imshow(np.log10(s["rho"]).T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="Greys",
                 interpolation="nearest")
        near = lost & (np.abs(d["t"] - t0) < 0.004)
        A.scatter(d["x"][near], d["y"][near], s=4, color="#D55E00", lw=0, alpha=0.7)
        A.scatter([x0], [y0], s=90, facecolor="none", edgecolor="#D55E00" if lt["fail"][ln] else "#0072B2", lw=1.8)
        hw = 0.16
        A.set_xlim(x0 - hw, x0 + hw); A.set_ylim(y0 - hw, y0 + hw)
        A.set_title("%s\nt = %.3f at (%.3f, %.3f): %s" % (
            (a.labels[r] if a.labels else typ), t0, x0, y0, ver), fontsize=9,
            color="#D55E00" if lt["fail"][ln] else "#0072B2", loc="left")
        A.set_xticks([]); A.set_yticks([])
        # the tube profiles against x/t
        xis = {}
        for nc in a.ncells:
            P_, tt_ = prof[nc]
            xis[nc] = (((np.arange(nc) + 0.5) / nc - 0.5) / tt_[r], P_)
        xi, P = xis[nc0]
        use_z = abs(UL[r, 6]) + abs(UR[r, 6]) > abs(UL[r, 5]) + abs(UR[r, 5])
        bt_key, vt_key = ("Bz", "vz") if use_z else ("By", "vy")
        panels = (
            ("density and gas pressure, over their left values", (("rho", "ρ", "#222222"), ("p", "p", "#E69F00")), True),
            ("velocity", (("vx", "v normal", "#009E73"), (vt_key, "v tangential", "#56B4E9")), False),
            ("tangential field", ((bt_key, "B tangential", "#D55E00"),), False),
        )
        for j, (title, series, logy) in enumerate(panels):
            B_ = ax[r, j + 1]
            for key, lab, col in series:
                ref = xis[nc0][1][key][0, r] if logy else 1.0
                for kk, nc in enumerate(a.ncells):
                    x_, P_ = xis[nc]
                    B_.plot(x_, P_[key][:, r] / ref, color=col, lw=(1.7, 1.0, 0.8)[min(kk, 2)],
                            alpha=(1.0, 0.55, 0.3)[min(kk, 2)], label=lab if kk == 0 else None)
            pass
            # characteristic speeds of the two states, as ticks at the top
            ymin, ymax = B_.get_ylim()
            for nm, xs in (("fast", (eig[r, 0], eigR[r, 3])), ("Alfvén", (la[r], raR[r])),
                           ("slow", (eig[r, 1], eigR[r, 2]))):
                for xv in xs:
                    B_.axvline(xv, color=SPEED[nm], lw=0.9, ls=":", alpha=0.9)
            B_.axvline(0.0, color=MUTED, lw=0.6)
            B_.set_xlim(min(eig[r, 0], -0.2) - 0.08, max(eigR[r, 3], 0.2) + 0.08)
            B_.set_title(title, fontsize=9.5, color=INK)
            B_.spines[["top", "right"]].set_visible(False)
            B_.tick_params(labelsize=8, colors=MUTED)
            B_.legend(frameon=False, fontsize=8, loc="best")
            if r == n - 1:
                B_.set_xlabel("x / t  (dotted: fast grey, Alfvén pink, slow blue speeds of L and R)",
                              fontsize=8.5, color=MUTED)
        if j == 2 and False:
            pass
        # (5) zoom: the tangential field where the slow and Alfven speeds sit
        Z = ax[r, 4]
        sl, sr_ = eig[r, 1], eigR[r, 2]
        cands = [c for c in (sl, sr_) if abs(c) > 0.05]
        c0 = 0.0 if not cands else (sl if abs(sl) > abs(sr_) else sr_)
        if abs(sl) < 0.05 and abs(sr_) < 0.05:
            c0, hw2 = 0.5 * (sl + sr_), 0.04
        else:
            hw2 = 0.12
        for kk, nc in enumerate(a.ncells):
            x_, P_ = xis[nc]
            Z.plot(x_, P_[bt_key][:, r], color="#D55E00", lw=(1.7, 1.0, 0.8)[min(kk, 2)],
                   alpha=(1.0, 0.55, 0.3)[min(kk, 2)], marker="." if kk == 0 else None, ms=2.5,
                   label="%d cells" % nc)
        for nm, xs in (("Alfvén", (la[r], raR[r])), ("slow", (eig[r, 1], eigR[r, 2]))):
            for xv in xs:
                Z.axvline(xv, color=SPEED[nm], lw=1.0, ls=":")
        Z.axvline(0.0, color=MUTED, lw=0.6)
        Z.axhline(0.0, color="#cccccc", lw=0.6)
        Z.set_xlim(c0 - hw2, c0 + hw2)
        Z.set_title("zoom: tangential field at the slow and Alfvén speeds", fontsize=9.5, color=INK)
        Z.spines[["top", "right"]].set_visible(False)
        Z.tick_params(labelsize=8, colors=MUTED)
        Z.legend(frameon=False, fontsize=7.5, loc="best")
        if r == n - 1:
            Z.set_xlabel("x / t", fontsize=8.5, color=MUTED)
        # the flux the run used against the tube's at x/t = 0
        i0 = np.argmin(np.abs(xi))
        st0 = {k_: torch.as_tensor(P[k_][i0:i0 + 1, r]) for k_ in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")}
        st0["eps"] = eos.eps__press_rho(st0["p"], st0["rho"])
        mk = lambda U: {k_: torch.as_tensor([v_]) for k_, v_ in zip(("rho", "vx", "vy", "vz", "By", "Bz"),
                                                                     (U[0], U[2], U[3], U[4], U[5], U[6]))}
        def prim(U):
            q = mk(U)
            v2 = U[2] ** 2 + U[3] ** 2 + U[4] ** 2
            eta = U[7] * U[2] + U[5] * U[3] + U[6] * U[4]
            b2 = (U[7] ** 2 + U[5] ** 2 + U[6] ** 2) * (1 - v2) + eta ** 2
            q["p"] = torch.as_tensor([U[1] - 0.5 * b2]); q["Bx"] = torch.as_tensor([U[7]])
            q["eps"] = eos.eps__press_rho(q["p"], q["rho"])
            return q
        try:
            Fh = hlld_flux(prim(UL[r]), prim(UR[r]), eos, idir=0)
            Fh = Fh[0] if isinstance(Fh, tuple) else Fh
            _, _, F0, _, _, _ = compute_srmhd_fluxes(st0, st0, eos, idir=0)
            keys = [k_ for k_ in F0 if k_ in Fh]
            num = max(float(abs(Fh[k_] - F0[k_]).max()) for k_ in keys)
            den = max(float(abs(F0[k_]).max()) for k_ in keys)
            ferr = num / den
        except Exception as e:                        # the flux comparison is a bonus
            ferr = np.nan
            print("flux comparison skipped for lane %d: %s" % (ln, e))
        ax[r, 1].text(0.02, 0.04, "HLLD flux vs tube at x/t = 0: %.1e (max rel.)" % ferr,
                      transform=ax[r, 1].transAxes, fontsize=8, color=INK)
        print("lane %d: %s %s t %.3f (%.3f, %.3f)  flux rel. diff %.2e" % (ln, typ, ver, t0, x0, y0, ferr))
    fig.suptitle("Faces of the four-quadrant 128² run and what their own Riemann problems build: lost ones "
                 "(orange titles) and solved ones nearby (blue)\n1D tube of each face's two states, HLLD + MC, "
                 "%s cells (thick to faint)" % " / ".join(str(c) for c in a.ncells), fontsize=11.5, color=INK)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
