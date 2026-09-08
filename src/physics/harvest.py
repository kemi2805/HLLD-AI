"""
Harvest Riemann problems out of a running simulation.

Every sweep of a 2D run hands the exact solver tens of thousands of genuine
interface states.  Two subsets of those are worth far more than anything a
random generator produces, and both are thrown away today:

**Solved after retries.**  A lane the ML seed missed and the retry ladder
rescued is a hard problem WITH a known exact answer.  It is in-distribution
by construction -- it is literally a state the rotor produced -- which is
precisely what the synthetic training sets are only ever approximating.  The
envelope work (docs/rotor_envelope.md) exists because guessing that
distribution went wrong once already; this removes the guess.

**Unsolved.**  The lanes that fell back to HLLD are the failure set, in the
regime that actually matters.  They are what a targeted retrain would aim
at, and what the failure plots should be drawn from.

Volume is the design constraint: a 256^2 rotor to t=0.4 is ~668M solves, so
recording everything is not an option.  Two knobs bound it -- ``stride``
(keep 1 in N eligible) and ``max_records`` (a hard cap per category, after
which recording stops rather than evicting, so what you keep is a clean
prefix of the run rather than a biased tail).  Defaults are deliberately
conservative.

The solved shards are written in the SAME layout as
``riemann_dataset.solutions_to_arrays`` -- ``U_L``/``U_R`` (N,8),
``zones`` (N,8,7), ``speeds`` (N,7), ``gamma`` -- so they can be merged with
``scripts/merge_chunks.py`` and fed straight to ``train_from_dataset`` with
no conversion step.  Only seven-wave lanes (FULL7, COPLANAR) are harvested
as solved: the degenerate classes carry a three-wave structure that does not
fit that schema, and silently writing a malformed zone table would poison a
training set in a way that is very hard to trace back.

States are recorded in the SOLVER's frame (normal, tangential-1,
tangential-2), not the grid frame, because that is the frame the solver was
handed and the only one in which the stored solution is self-consistent.
"""
from __future__ import annotations

import os
import time

import numpy as np

# reasons a lane did not yield a usable exact flux
R_NOT_CONVERGED = 0
R_RAY_FAILED = 1
R_UNPHYSICAL = 2
R_REASONS = {R_NOT_CONVERGED: "not_converged",
             R_RAY_FAILED: "ray_failed",
             R_UNPHYSICAL: "unphysical"}


class Harvester:
    """Buffers solved/unsolved interface problems and flushes them to npz.

    Parameters
    ----------
    out_dir        directory for the shards
    gamma          adiabatic index, stored with every shard
    stride         keep 1 in ``stride`` eligible records
    max_solved     hard cap on solved records (then stops recording)
    max_unsolved   hard cap on unsolved records
    only_retried   if True (default) record only lanes the ML seed missed and
                   the retry ladder rescued -- the hard ones.  Recording the
                   lanes that converged first time would bury those under the
                   easy majority, which is the same mistake as sampling a
                   training set uniformly from a distribution whose interesting
                   part is a thin tail.
    flush_every    write a shard once a buffer reaches this many rows
    rank           MPI rank, folded into filenames so ranks cannot collide
    """

    def __init__(self, out_dir, gamma, *, stride=1, max_solved=200_000,
                 max_unsolved=200_000, only_retried=True, flush_every=20_000,
                 rank=0):
        self.out_dir = out_dir
        self.gamma = float(gamma)
        self.stride = max(1, int(stride))
        self.max_solved = int(max_solved)
        self.max_unsolved = int(max_unsolved)
        self.only_retried = bool(only_retried)
        self.flush_every = int(flush_every)
        self.rank = int(rank)

        self._solved = []          # list of (U_L, U_R, zones, speeds, attempts)
        self._unsolved = []        # list of (U_L, U_R, reason, attempts, cls)
        # per-sweep coverage: which interfaces were attempted and which came
        # out exact.  Everything else kept the HLLD flux by the gates; an
        # attempted-but-not-exact interface kept it by fallback (its reason is
        # in the unsolved shards).  Packed bits, ~1 KB per sweep at 64^2.
        self._coverage = []        # list of dict(idir, n, attempted, exact)
        self.n_sweeps = 0
        self.n_solved = 0
        self.n_unsolved = 0
        self._seen = 0             # eligible lanes seen, for the stride
        self._shard = 0
        os.makedirs(out_dir, exist_ok=True)

    # ── recording ────────────────────────────────────────────────────────
    def _subsample(self, idx):
        """Apply the stride without letting it lock onto a lane index.

        Striding on a per-call counter rather than on the lane index matters:
        interface index correlates with position in the grid, so a fixed
        ``idx % stride`` would sample the same spatial stripe every sweep and
        miss whole regions of the flow.
        """
        if self.stride == 1:
            return idx
        off = self._seen % self.stride
        self._seen += idx.size
        return idx[(np.arange(idx.size) + off) % self.stride == 0]

    def record(self, left, right, Bn, *, cls, converged, zones, VsLv, VsRv,
               attempts, accepted, seven_wave, source=0, ray_ok=None):
        """Record one batch.

        ``left``/``right`` are 7-lists of per-lane arrays in the solver frame,
        ``zones`` the six R2..R7 zone states, ``accepted`` the lanes that
        passed every gate (converged, ray resolved, state physical).

        ``source`` tags which solver produced the rows: 0 = the seven-wave
        Newton, 1 = the five-wave planar solver.  Both write exactly the same
        schema -- a planar answer IS a seven-wave solution whose two
        rotational discontinuities have zero strength -- but the planar rows
        all carry ``phi_L = phi_R = 0`` exactly, so a trainer may want to
        weight or ablate them.  Scalar or per-lane.

        ``ray_ok`` separates a lane that converged but produced no usable ray
        from one whose state came out unphysical.  Without it both are
        recorded as unphysical, which is what the code did before this
        argument existed even though ``R_RAY_FAILED`` was defined.

        MAY BE CALLED MORE THAN ONCE PER SWEEP, but only on DISJOINT lane
        sets: the unsolved branch fires on every call, so a second call must
        be restricted to the lanes it actually owns or it re-records the
        first call's successes as failures.
        """
        n = Bn.shape[0]
        att = np.asarray(attempts)
        src = np.broadcast_to(np.asarray(source, dtype=np.int8), (n,))

        # ── solved ───────────────────────────────────────────────────────
        if self.n_solved < self.max_solved:
            ok = accepted & seven_wave
            if self.only_retried:
                ok = ok & (att > 1)
            idx = self._subsample(np.flatnonzero(ok))
            room = self.max_solved - self.n_solved
            idx = idx[:room]
            if idx.size:
                self._solved.append(self._pack_solved(
                    left, right, Bn, zones, VsLv, VsRv, att, idx, src))
                self.n_solved += idx.size

        # ── unsolved ─────────────────────────────────────────────────────
        if self.n_unsolved < self.max_unsolved:
            bad = ~accepted
            if bad.any():
                # a lane that converged but produced no usable ray is a
                # different failure from one whose state came out unphysical;
                # keeping them apart is what makes the failure plots readable
                reason = np.full(n, R_UNPHYSICAL, dtype=np.int8)
                if ray_ok is not None:
                    reason[np.asarray(converged) & ~np.asarray(ray_ok)] = R_RAY_FAILED
                reason[~np.asarray(converged)] = R_NOT_CONVERGED
                idx = self._subsample(np.flatnonzero(bad))
                idx = idx[:self.max_unsolved - self.n_unsolved]
                if idx.size:
                    self._unsolved.append(dict(
                        U_L=self._u8(left, Bn, idx),
                        U_R=self._u8(right, Bn, idx),
                        reason=reason[idx],
                        attempts=att[idx].astype(np.int16),
                        cls=np.asarray(cls)[idx].astype(np.int8)))
                    self.n_unsolved += idx.size

        if (sum(a["U_L"].shape[0] for a in self._solved) >= self.flush_every
                or sum(a["U_L"].shape[0] for a in self._unsolved)
                >= self.flush_every):
            self.flush()

    @staticmethod
    def _u8(state, Bn, idx):
        """(n,8) row [rho,P,vx,vy,vz,By,Bz,Bx] -- solutions_to_arrays layout."""
        return np.stack([np.asarray(c)[idx] for c in state]
                        + [np.asarray(Bn)[idx]], axis=1)

    def _pack_solved(self, left, right, Bn, zones, VsLv, VsRv, att, idx,
                     src=None):
        k = idx.size
        U_L = self._u8(left, Bn, idx)
        U_R = self._u8(right, Bn, idx)

        # zones R1..R8 = left, R2..R7, right -- the schema the generator and
        # the trainer both assume
        Z = np.empty((k, 8, 7))
        for j in range(7):
            Z[:, 0, j] = np.asarray(left[j])[idx]
            Z[:, 7, j] = np.asarray(right[j])[idx]
        for z in range(6):
            for j in range(7):
                Z[:, z + 1, j] = np.asarray(zones[z][j])[idx]

        # speeds in SPATIAL order (LF, LA, LS, CD, RS, RA, RF).  VsRv is
        # stored outward-from-the-contact (RF, RA, RS), so it reverses here --
        # getting this backwards writes a plausible-looking but wrong table.
        vxc = 0.5 * (Z[:, 3, 2] + Z[:, 4, 2])          # R4.vx, R5.vx
        S = np.stack([np.asarray(VsLv[0])[idx], np.asarray(VsLv[1])[idx],
                      np.asarray(VsLv[2])[idx], vxc,
                      np.asarray(VsRv[2])[idx], np.asarray(VsRv[1])[idx],
                      np.asarray(VsRv[0])[idx]], axis=1)
        if src is None:
            src = np.zeros(len(att), dtype=np.int8)
        return dict(U_L=U_L, U_R=U_R, zones=Z, speeds=S,
                    attempts=att[idx].astype(np.int16),
                    source=np.asarray(src)[idx].astype(np.int8))

    def record_coverage(self, idir, n_interfaces, attempted_idx, exact_idx):
        """One sweep's coverage: flat interface indices that were attempted
        and that came out exact (``exact_idx`` is a subset of ``attempted_idx``).
        Not subject to the stride or the caps -- it is the complete map."""
        att = np.zeros(int(n_interfaces), dtype=np.bool_)
        ex = np.zeros(int(n_interfaces), dtype=np.bool_)
        att[np.asarray(attempted_idx, dtype=np.int64)] = True
        ex[np.asarray(exact_idx, dtype=np.int64)] = True
        self._coverage.append(dict(sweep=self.n_sweeps, idir=int(idir),
                                   n=int(n_interfaces),
                                   attempted=np.packbits(att), exact=np.packbits(ex)))
        self.n_sweeps += 1

    # ── output ───────────────────────────────────────────────────────────
    def flush(self):
        stamp = f"r{self.rank:03d}_{self._shard:05d}"
        for name, buf in (("solved", self._solved),
                          ("unsolved", self._unsolved)):
            if not buf:
                continue
            merged = {k: np.concatenate([b[k] for b in buf], axis=0)
                      for k in buf[0]}
            merged["gamma"] = self.gamma
            path = os.path.join(self.out_dir, f"{name}_{stamp}.npz")
            np.savez_compressed(path, **merged)
            buf.clear()
        if self._coverage:
            c = self._coverage
            np.savez_compressed(
                os.path.join(self.out_dir, f"coverage_{stamp}.npz"),
                sweep=np.array([d["sweep"] for d in c], dtype=np.int64),
                idir=np.array([d["idir"] for d in c], dtype=np.int8),
                n=np.array([d["n"] for d in c], dtype=np.int64),
                attempted=np.stack([d["attempted"] for d in c], axis=0),
                exact=np.stack([d["exact"] for d in c], axis=0))
            c.clear()
        self._shard += 1

    def summary(self):
        return dict(harvest_solved=self.n_solved,
                    harvest_unsolved=self.n_unsolved,
                    harvest_sweeps=self.n_sweeps,
                    harvest_dir=self.out_dir)

    def close(self):
        self.flush()
        return self.summary()
