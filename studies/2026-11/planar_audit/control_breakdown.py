import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
np.set_printoptions(linewidth=200, precision=4, suppress=True)
for tag in ("all0", "all1"):
    a = WC.admissibility(z[tag + "_U_L"], z[tag + "_U_R"], z[tag + "_zones"], z[tag + "_speeds"], G)
    S = z[tag + "_speeds"]
    n = S.shape[0]
    print("==", tag, "rows", n)
    anybad = np.zeros(n, bool)
    for w in ("LF", "LS", "RS", "RF"):
        sh = a[w + "_solver_kind"] == 1
        K = a[w + "_K"]
        M = a[w + "_lax_margins"]
        tags = a[w + "_lax_tags"]
        ent = sh & (K < 1 - 1e-6)
        print("  %s solver-shocks %d | entropy decrease (K<1-1e-6) %d" % (w, sh.sum(), ent.sum()))
        for j, t in enumerate(tags):
            for thr in (1e-6, 1e-4, 1e-2):
                m = sh & (M[j] < -thr)
                if m.any():
                    print("     lax %-18s < -%g : %4d" % (t, thr, m.sum()))
        anybad |= ent | (sh & (M.min(axis=0) < -1e-4))
    # crossing of slow vs fast (the zero rotation between them exempts nothing)
    lf_right = np.where(a["LF_solver_kind"] == 2, a["LF_fan_tail"], S[:, 0])
    ls_left = np.where(a["LS_solver_kind"] == 2, a["LS_fan_head"], S[:, 2])
    rs_right = np.where(a["RS_solver_kind"] == 2, a["RS_fan_head"], S[:, 4])
    rf_left = np.where(a["RF_solver_kind"] == 2, a["RF_fan_tail"], S[:, 6])
    nzL = (a["LF_solver_kind"] > 0) & (a["LS_solver_kind"] > 0)
    nzR = (a["RF_solver_kind"] > 0) & (a["RS_solver_kind"] > 0)
    for thr in (1e-6, 1e-4, 1e-2):
        print("  LS left of LF by >%g: %d   RS right of RF by >%g: %d" % (
            thr, (nzL & (ls_left < lf_right - thr)).sum(), thr, (nzR & (rs_right > rf_left + thr)).sum()))
    print("  rows with entropy-decreasing shock or Lax < -1e-4:", anybad.sum())
