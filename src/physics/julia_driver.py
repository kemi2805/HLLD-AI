"""HLLD's whole exact flux in Julia (``RMHD_JULIA=driver``, 2026-11-09, user:
"start with 1-4").

Where ``RMHD_JULIA=1`` (julia_chain.py) hands Julia only the attempted lanes
of a sweep, and Python keeps HLLD, the gates, the canonical orientation, the
seven-wave seed networks and the flux assembly, this hands Julia the sweep's
face states as they come out of the reconstruction and Julia answers all of
it (julia_proto/serve_driver.jl, src/driver.jl): the canonical orientation,
HLLD (the fallback flux and the star-pressure hint), the bad / weak / upwind
gates, the linearised flux, the seeds (the run's checkpoints, exported once to
``RMHD_JULIA_NETS`` by julia_proto's export_nets.py), the exact chain, the flux
of the state at xi = 0, and the map back.  Python only packs and unpacks.

The same environment as julia_chain.py (RMHD_JULIA_EXE, RMHD_JULIA_PROTO,
RMHD_JULIA_THREADS, RMHD_JL_FAN) plus RMHD_JULIA_NETS, the directory for the
exported networks (default <proto>/data_nets).  The ladder's configuration
only (``julia_chain.preflight``), with RMHD_CANONICAL on, no harvest and no
merged sweeps.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

MAGIC_D = 50004343.0
NOUT = 25
_PROC = None
_LOCK = threading.RLock()
_STATS = dict(calls=0, faces=0, seconds=0.0)
_PK = ("rho", "vx", "vy", "vz", "p", "eps", "Bx", "By", "Bz")
_CK = ("D", "tau", "Sx", "Sy", "Sz", "Bx", "By", "Bz")
# the driver's codes (driver.jl FaceAnswer)
G_ATTEMPTED, G_BAD, G_WEAK, G_UPWIND = 0, 1, 2, 3
K_HLLD, K_HLLE, K_LINEAR, K_EXACT = 0, 1, 2, 3


def _read(p, n):
    need = 8 * n
    buf = bytearray()
    while len(buf) < need:
        chunk = p.stdout.read(need - len(buf))
        if not chunk:
            raise RuntimeError("the Julia driver server died (see its stderr above)")
        buf += chunk
    return np.frombuffer(bytes(buf), dtype="<f8")


def _nets():
    """The run's seed networks exported for Julia: (primary, extras...)."""
    from . import exact_flux as EF
    import rmhd.paths as _rp
    proto = os.environ["RMHD_JULIA_PROTO"]
    outdir = os.environ.get("RMHD_JULIA_NETS", os.path.join(proto, "data_nets"))
    os.makedirs(outdir, exist_ok=True)
    sys.path.insert(0, os.path.join(proto, "port64", "standalone"))
    import export_nets
    paths = []
    for ck in [EF._DEFAULT_CKPT] + list(EF._EXTRA_CKPTS):
        src = _rp.resolve(ck)
        dst = os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + ".bin")
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(os.path.realpath(src)):
            export_nets.export(src, outdir)
        paths.append(dst)
    return paths


def _start():
    exe = os.environ.get("RMHD_JULIA_EXE", "julia")
    proto = os.environ["RMHD_JULIA_PROTO"]
    threads = os.environ.get("RMHD_JULIA_THREADS", "64")
    cmd = [exe, "-t", threads, "--project=" + proto, "-O3", os.path.join(proto, "serve_driver.jl")] + _nets()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    h = _read(p, 2)
    if h[0] != MAGIC_D:
        raise RuntimeError("the Julia driver server answered %r instead of its ready line" % (h,))
    print("julia driver: server up in %.1f s (%d threads)" % (time.perf_counter() - t0, int(h[1])), flush=True)
    atexit.register(_stop)
    return p


def _stop():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.stdin.write(np.array([MAGIC_D] + [0.0] * 11, dtype="<f8").tobytes())
            _PROC.stdin.flush()
            _PROC.wait(timeout=60)
        except Exception:
            _PROC.kill()
    if _STATS["calls"]:
        print("julia driver: %d sweeps, %d faces, %.1f s in Julia" % (
            _STATS["calls"], _STATS["faces"], _STATS["seconds"]), flush=True)
        _STATS.update(calls=0, faces=0, seconds=0.0)
    _PROC = None


def _preflight(EF, *, tau_bt, harvester, rendezvous, model, scaler):
    from . import julia_chain as _JC
    _JC.preflight(EF)
    bad = []
    if not EF._CANONICAL:
        bad.append("RMHD_CANONICAL must be on (the driver solves every face as its canonical image)")
    if tau_bt != 1e-9:
        bad.append("tau_bt %r (the chain hard-codes 1e-9)" % tau_bt)
    if harvester is not None:
        bad.append("no harvest in the driver mode yet")
    if rendezvous is not None:
        bad.append("RMHD_MERGE_SWEEPS must be 0 (one sweep per call)")
    if model is not None or scaler is not None:
        bad.append("the seeds come from RMHD_ML_CKPT / RMHD_ML_CKPTS, not a model argument")
    if EF._WEAK_FLUX not in ("linear", "hlld"):
        bad.append("RMHD_WEAK_FLUX=%r" % EF._WEAK_FLUX)
    if bad:
        raise RuntimeError("RMHD_JULIA=driver: " + "; ".join(bad))


def flux(sL, sR, eos, idir, *, tau_weak, tau_bt, accuracy, max_iter, n_retries, harvester=None,
         rendezvous=None, model=None, scaler=None):
    """exact_flux_batched's contract: (F, U, p_star) for every face, LAST_DIAG
    filled with what the run's log reads."""
    global _PROC
    from . import exact_flux as EF
    from . import linearised as LN
    from . import canonical as CN
    from .hlld import LAST_DIAG
    _preflight(EF, tau_bt=tau_bt, harvester=harvester, rendezvous=rendezvous, model=model, scaler=scaler)
    gamma = EF._check_eos(eos)
    shape = sL["rho"].shape
    dt = sL["rho"].dtype
    n = sL["rho"].numel()
    pack = lambda s: np.ascontiguousarray(np.stack(
        [s[k].detach().reshape(-1).cpu().numpy().astype(float) for k in _PK], axis=1))
    a, b = pack(sL), pack(sR)
    flags = (1 if CN._POS_ZERO else 0) | (2 if EF._WEAK_FLUX == "linear" else 0) | (4 if EF._UPWIND_SKIP else 0)
    hdr = np.array([MAGIC_D, n, idir, gamma, tau_weak, accuracy, max_iter, n_retries, LN._ZERO_TOL,
                    EF._WEAK_FALLBACK_TAU, flags, 0.0], dtype="<f8")
    t0 = time.perf_counter()
    with _LOCK:
        if _PROC is None:
            _PROC = _start()
        _PROC.stdin.write(hdr.tobytes() + a.tobytes() + b.tobytes())
        _PROC.stdin.flush()
        h = _read(_PROC, 2)
        if h[0] != MAGIC_D or int(h[1]) != n:
            raise RuntimeError("the Julia driver answered %r for %d faces" % (h, n))
        res = _read(_PROC, NOUT * n).reshape(n, NOUT)
    wall = time.perf_counter() - t0
    _STATS["calls"] += 1; _STATS["faces"] += n; _STATS["seconds"] += wall
    gate = res[:, 17].astype(int)
    if (gate < 0).any():
        # a face Julia could not answer: production's HLLD there (never a hole)
        from .hlld import hlld_flux
        i = np.flatnonzero(gate < 0)
        Fh, Uh, ph = hlld_flux({k: v.reshape(-1)[i] for k, v in sL.items()},
                               {k: v.reshape(-1)[i] for k, v in sR.items()}, eos, idir=idir)
        for j, k in enumerate(_CK):
            res[i, j] = Fh[k].numpy(); res[i, 8 + j] = Uh[k].numpy()
        res[i, 16] = ph.numpy(); res[i, 18] = K_HLLD
    T = lambda c: torch.as_tensor(np.ascontiguousarray(c), dtype=dt).reshape(shape)
    F = {k: T(res[:, j]) for j, k in enumerate(_CK)}
    U = {k: T(res[:, 8 + j]) for j, k in enumerate(_CK)}
    p_star = T(res[:, 16])
    kind = res[:, 18].astype(int)
    rung = res[:, 19].astype(int)
    att = gate == G_ATTEMPTED
    exact = kind == K_EXACT
    n_att = int(att.sum()); n_exact = int(exact.sum())
    LAST_DIAG.update(
        solver="exact-batched-julia-driver", idir=idir, n_interfaces=n,
        n_bad=int((gate == G_BAD).sum()), n_weak_gate=int((gate == G_WEAK).sum()),
        n_weak_linear=int(((gate == G_WEAK) & (kind == K_LINEAR)).sum()),
        n_upwind_skip=int((gate == G_UPWIND).sum()), n_compound_skip=0,
        n_attempted=n_att, n_exact=n_exact, n_verified=n_exact,
        n_linear_fallback=int((att & (kind == K_LINEAR)).sum()),
        n_hlld_fallback=n - n_exact, frac_hlld_fallback=(n - n_exact) / max(n, 1),
        frac_exact=n_exact / max(n, 1),
        # HLLD's own fallback to HLLE (production's hlld_flux reports it here)
        frac_hlle_fallback=float((res[:, 23] != 0).sum()) / max(n, 1),
        exact_mask=torch.as_tensor(exact),
        weak_linear_mask=torch.as_tensor((gate == G_WEAK) & (kind == K_LINEAR)),
        canonical_g=res[:, 22].astype(np.int8), n_canonical_moved=int((res[:, 22] != 0).sum()),
        rung=rung.astype(np.int8), julia_wall=wall, julia_face_seconds=float(res[:, 24].sum()))
    if EF._RUNG_LOG:
        from . import julia_chain as _JC
        a_ = att
        _JC.log_rungs(EF._RUNG_LOG, idir, n, n_att, a_ & (rung == 1), a_ & (rung == 5), a_ & (rung == 2),
                      a_ & (rung == 3), a_ & (rung == 4), a_ & (kind == K_LINEAR), wall)
    return F, U, p_star
