"""Shared loading and feature matrix for the anatomy study (2026-11-04)."""
import numpy as np

NAMES = ["seven, first try", "seven, after retry", "planar five-wave", "four-unknown", "lost to HLLD"]
SHORT = ["seven-1", "seven-retry", "planar5", "planar4", "lost"]


def load(path="data/sample.npz"):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    oc, at = d["outcome"], d["attempts"]
    d["group"] = np.where(oc == 1, np.where(at > 1, 1, 0), np.where(oc == 2, 2, np.where(oc == 3, 3, 4)))
    d["gapAS"] = np.minimum(d["f_gapAS_L"], d["f_gapAS_R"]).astype(float)
    d["gapFA"] = np.minimum(d["f_gapFA_L"], d["f_gapFA_R"]).astype(float)
    return d


def lg(x, floor):
    return np.log10(np.maximum(np.abs(x.astype(float)), floor))


def matrix(d):
    """State-only features: available for every face, lost or not."""
    cols = {
        "log10 slow-Alfven gap": lg(d["gapAS"], 1e-9),
        "log10 fast-Alfven gap": lg(d["gapFA"], 1e-9),
        "normal-field fraction (max)": d["f_fn_max"].astype(float),
        "normal-field fraction (min)": d["f_fn_min"].astype(float),
        "log10 sigma (max)": lg(d["f_sigma_max"], 1e-9),
        "log10 beta (min)": lg(d["f_beta_min"], 1e-9),
        "tangential Lorentz factor": np.minimum(d["f_Wt"].astype(float), 50),
        "Lorentz factor (max)": np.minimum(d["f_W_max"].astype(float), 50),
        "log10 jump": lg(d["f_jump"], 1e-12),
        "log10 density ratio": d["f_log_rho_ratio"].astype(float),
        "log10 pressure ratio": d["f_log_p_ratio"].astype(float),
        "normal velocity jump": d["f_dvn"].astype(float),
        "tangential velocity jump": d["f_shear"].astype(float),
        "field reversal": d["f_reversal"].astype(float),
        "log10 min|Bt|/|Bn|": lg(d["f_minBt_over_Bn"], 1e-9),
    }
    X = np.stack(list(cols.values()), axis=1)
    X = np.where(np.isfinite(X), X, 0.0)
    return X, list(cols)
