#!/usr/bin/env python
"""Admissibility of every harvested exact answer (2026-11-03, planar acceptance
study): the 4Q 64^2 run's harvest (it1/all.npz = every exact face of
r2d_64_exact_ssretry, seven-wave + planar five-wave) and the perturbed set.

Per row, per magnetosonic wave, by the solvers' own shock/fan predicates:
entropy ratio K_b/K_a (K = p_gas/rho^gamma), the Lax margins (wavecheck.py),
and the order of each slow wave against the fast wave on its side with NO
exemption for a zero-strength rotation between them.

    python admiss_all.py RETRAIN_DIR OUT.npz OUT.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/rafast/miler/scratch/r2d_retrain/planar_audit")
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import wavecheck as WC        # noqa: E402

G = 5.0 / 3.0


def keys16(UL, UR):
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1), dtype=np.float64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def flags(UL, UR, Z, S, chunk=200_000):
    n = UL.shape[0]
    out = {k: np.zeros(n, bool) for k in ("ent_LS", "ent_RS", "ent_F", "lax_LS", "lax_RS", "lax_F",
                                          "inter_LS", "inter_RS", "cross_L", "cross_R")}
    out.update(Kmin=np.ones(n), laxmin_slow=np.zeros(n), cross_amt=np.zeros(n))
    for i in range(0, n, chunk):
        s = slice(i, i + chunk)
        a = WC.admissibility(UL[s], UR[s], Z[s], S[s], G)
        Sp = S[s]
        for w in ("LF", "LS", "RS", "RF"):
            sh = a[w + "_solver_kind"] == 1
            ent = sh & (a[w + "_K"] < 1 - 1e-6)
            lax = sh & (a[w + "_lax_min"] < -1e-4)
            if w in ("LS", "RS"):
                out["ent_" + w][s] = ent
                out["lax_" + w][s] = lax
                # beyond the Alfven speed ahead (the MARGIN band): margin 2 (LS) / 3 (RS)
                j = 2 if w == "LS" else 3
                out["inter_" + w][s] = sh & (a[w + "_lax_margins"][j] < -1e-6)
                out["laxmin_slow"][s] = np.minimum(out["laxmin_slow"][s], np.where(sh, a[w + "_lax_min"], 0.0))
            else:
                out["ent_F"][s] |= ent
                out["lax_F"][s] |= lax
            out["Kmin"][s] = np.minimum(out["Kmin"][s], np.where(sh, a[w + "_K"], 1.0))
        lf_r = np.where(a["LF_solver_kind"] == 2, a["LF_fan_tail"], Sp[:, 0])
        ls_l = np.where(a["LS_solver_kind"] == 2, a["LS_fan_head"], Sp[:, 2])
        rs_r = np.where(a["RS_solver_kind"] == 2, a["RS_fan_head"], Sp[:, 4])
        rf_l = np.where(a["RF_solver_kind"] == 2, a["RF_fan_tail"], Sp[:, 6])
        nzL = (a["LF_solver_kind"] > 0) & (a["LS_solver_kind"] > 0)
        nzR = (a["RF_solver_kind"] > 0) & (a["RS_solver_kind"] > 0)
        out["cross_L"][s] = nzL & (ls_l < lf_r - 1e-6)
        out["cross_R"][s] = nzR & (rs_r > rf_l + 1e-6)
        out["cross_amt"][s] = np.maximum(np.where(nzL, lf_r - ls_l, 0.0), np.where(nzR, rs_r - rf_l, 0.0))
    out["entropy"] = out["ent_LS"] | out["ent_RS"] | out["ent_F"]
    out["lax"] = out["lax_LS"] | out["lax_RS"] | out["lax_F"]
    out["cross"] = out["cross_L"] | out["cross_R"]
    out["any"] = out["entropy"] | out["lax"] | out["cross"]
    return out


def main():
    D, OUT, OJ = sys.argv[1], sys.argv[2], sys.argv[3]
    summary = {}
    save = {}
    for f in ("it1/all.npz", "it2/perturb.npz"):
        z = np.load(D + "/" + f)
        UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
        Z, S = np.asarray(z["zones"], float), np.asarray(z["speeds"], float)
        src = np.asarray(z["source"]) if "source" in z.files else np.full(UL.shape[0], -1)
        t = np.asarray(z["t"], float) if "t" in z.files else np.full(UL.shape[0], np.nan)
        k = keys16(UL, UR)
        fl = flags(UL, UR, Z, S)
        r = {}
        for sname, m in (("all", np.ones(UL.shape[0], bool)), ("seven", src == 0), ("planar5", src == 1)):
            if not m.any():
                continue
            e = dict(rows=int(m.sum()), problems=int(np.unique(k[m]).size))
            for key in ("ent_LS", "ent_RS", "ent_F", "lax_LS", "lax_RS", "lax_F", "inter_LS", "inter_RS",
                        "cross_L", "cross_R", "entropy", "lax", "cross", "any"):
                e[key] = int((fl[key] & m).sum())
            e["any_problems"] = int(np.unique(k[fl["any"] & m]).size)
            e["entropy_or_cross_problems"] = int(np.unique(k[(fl["entropy"] | fl["cross"]) & m]).size)
            r[sname] = e
        summary[f] = r
        print(f, json.dumps(r, indent=1), flush=True)
        idx = np.flatnonzero(fl["any"])
        tag = f.split("/")[1].split(".")[0]
        save[tag + "_idx"] = idx
        save[tag + "_src"] = src[idx]
        save[tag + "_t"] = t[idx]
        for key in ("entropy", "lax", "cross", "inter_LS", "inter_RS", "Kmin", "laxmin_slow", "cross_amt"):
            save[tag + "_" + key] = fl[key][idx]
        for key in ("U_L", "U_R", "zones", "speeds"):
            save[tag + "_" + key] = np.asarray(z[key])[idx]
    np.savez_compressed(OUT, **save)
    json.dump(summary, open(OJ, "w"), indent=1)
    print("wrote", OUT, OJ)


if __name__ == "__main__":
    main()
