"""
Record the interface states a run actually feeds to the Riemann solver.

Why this exists
---------------
The ML surrogate predicts the HLLD star pressure ``p*``, and it is trained on
randomly sampled Riemann problems.  Those samples are only useful if they
cover the states the target problem really produces.  The existing
configuration was sampled for 1D shock tubes (``W in [1,2]``,
``p in [0.1,1]``) and does not come close to covering the magnetic rotor.

Rather than guess new ranges, measure them: the validation programme already
requires a plain-HLLD rotor run, so this accumulator rides along on it and
records the actual left/right interface states.  The measured envelope then
defines the sampling ranges for the retrain.

What is recorded
----------------
For a subsample of interfaces, per side, in the solver's own
normal/transverse decomposition (the same one the network's features use):

    log10 rho, log10 p, W, |B_t|            for L and R
    |B_n|                                   (single-valued at a face)
    cos(angle between the transverse v and B vectors)

The relative transverse v-B angle is included deliberately.  The current
15-feature vector carries only the magnitudes ``|v_t|`` and ``|B_t|`` and not
their relative orientation, so two states with identical features can have
different ``p*``.  Recording the angle here makes it possible to check
whether the rotor actually populates a wide range of it -- i.e. whether that
missing feature is a real error floor for this problem or a non-issue.

Subsampling (rather than histogramming) keeps the full JOINT distribution,
which is what a training sampler needs; marginal ranges alone would lose the
correlations.
"""

from __future__ import annotations

import numpy as np
import torch

FIELDS = ("lrho_L", "lp_L", "W_L", "Bt_L",
          "lrho_R", "lp_R", "W_R", "Bt_R",
          "Bn", "cos_vB", "sigma")


class EnvelopeRecorder:
    """Accumulates a subsample of interface states over a run."""

    def __init__(self, n_per_call: int = 2000, every: int = 5, seed: int = 0):
        self.n_per_call = n_per_call
        self.every = every
        self.rng = np.random.default_rng(seed)
        self.rows: list[np.ndarray] = []
        self._calls = 0

    def record(self, L: dict, R: dict, axis: int, eos) -> None:
        """Record one sweep's interface states (flat or 2D dicts both fine)."""
        self._calls += 1
        if (self._calls - 1) % self.every:
            return

        n = ["Bx", "By", "Bz"][axis]
        t1 = ["Bx", "By", "Bz"][(axis + 1) % 3]
        t2 = ["Bx", "By", "Bz"][(axis + 2) % 3]
        vn = ["vx", "vy", "vz"][axis]
        vt1 = ["vx", "vy", "vz"][(axis + 1) % 3]
        vt2 = ["vx", "vy", "vz"][(axis + 2) % 3]

        flat = {k: v.reshape(-1) for k, v in L.items()}
        flatR = {k: v.reshape(-1) for k, v in R.items()}
        ntot = flat["rho"].numel()
        k = min(self.n_per_call, ntot)
        idx = torch.from_numpy(
            self.rng.choice(ntot, size=k, replace=False).astype(np.int64))

        def take(d, key):
            return d[key][idx].double()

        out = {}
        for tag, d in (("L", flat), ("R", flatR)):
            rho, p = take(d, "rho"), take(d, "p")
            v2 = take(d, "vx") ** 2 + take(d, "vy") ** 2 + take(d, "vz") ** 2
            W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-16))
            Bt = torch.sqrt(take(d, t1) ** 2 + take(d, t2) ** 2)
            out[f"lrho_{tag}"] = torch.log10(torch.clamp(rho, min=1e-300))
            out[f"lp_{tag}"] = torch.log10(torch.clamp(p, min=1e-300))
            out[f"W_{tag}"] = W
            out[f"Bt_{tag}"] = Bt

        out["Bn"] = take(flat, n).abs()

        # relative orientation of the transverse velocity and field (left side)
        vt = torch.stack([take(flat, vt1), take(flat, vt2)], dim=1)
        Bt = torch.stack([take(flat, t1), take(flat, t2)], dim=1)
        nv = vt.norm(dim=1).clamp(min=1e-30)
        nb = Bt.norm(dim=1).clamp(min=1e-30)
        out["cos_vB"] = ((vt * Bt).sum(dim=1) / (nv * nb)).clamp(-1.0, 1.0)

        # magnetization sigma = b^2 / (rho h), the strongest known predictor of
        # solver difficulty in the companion exact-solver study
        rho = take(flat, "rho")
        pg = take(flat, "p")
        v2 = take(flat, "vx") ** 2 + take(flat, "vy") ** 2 + take(flat, "vz") ** 2
        W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-16))
        Bvec = torch.stack([take(flat, "Bx"), take(flat, "By"),
                            take(flat, "Bz")], dim=1)
        vvec = torch.stack([take(flat, "vx"), take(flat, "vy"),
                            take(flat, "vz")], dim=1)
        B2 = (Bvec ** 2).sum(dim=1)
        vB = (vvec * Bvec).sum(dim=1)
        b2 = B2 / W ** 2 + vB ** 2
        eps = take(flat, "eps")
        h = 1.0 + eps + pg / rho.clamp(min=1e-300)
        out["sigma"] = b2 / (rho * h).clamp(min=1e-300)

        self.rows.append(
            np.stack([out[f].numpy() for f in FIELDS], axis=1))

    def save(self, path: str, **meta) -> dict:
        """Write the samples and return a summary of the measured envelope."""
        data = (np.concatenate(self.rows, axis=0) if self.rows
                else np.zeros((0, len(FIELDS))))
        np.savez_compressed(path, samples=data, fields=np.array(FIELDS),
                            **meta)
        return summarize(data)


def summarize(data: np.ndarray, qlo: float = 0.001, qhi: float = 0.999) -> dict:
    """Per-field min / quantiles / max of a recorded envelope."""
    out = {}
    for i, f in enumerate(FIELDS):
        col = data[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        out[f] = dict(min=float(col.min()), max=float(col.max()),
                      qlo=float(np.quantile(col, qlo)),
                      qhi=float(np.quantile(col, qhi)),
                      median=float(np.median(col)))
    return out


def format_summary(s: dict) -> str:
    lines = [f"{'field':>8} {'min':>10} {'0.1%':>10} {'median':>10} "
             f"{'99.9%':>10} {'max':>10}"]
    for f, d in s.items():
        lines.append(f"{f:>8} {d['min']:10.3f} {d['qlo']:10.3f} "
                     f"{d['median']:10.3f} {d['qhi']:10.3f} {d['max']:10.3f}")
    return "\n".join(lines)
