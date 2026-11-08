"""
The planar warm-start network at inference (``RMHD_COPLANAR7_NET``, off by
default; coplanar7_b uses it).

For the late rung's lanes it predicts, from the face's two states:
  * the four planar unknowns in either steering: ``u_S`` (both slow waves
    to the signed field at the contact) and ``u_P`` (both to the total
    pressure at the contact);
  * which steering answers (``pressure``, one flag per lane: the late rung
    steers both slow waves the same way);
  * the rotation in front of each slow wave (``rot``, (2, n)).

The network was trained outside this package (claude_scratch/r2d_retrain/
fixes/planar_ml: planar_data.py, planar_train.py; v2a, calea job 84559,
2026-11-08). Its training rows were the late rung's, the four-unknown
solver's and the five-wave solver's answers of the 4Q, Orszag-Tang and rotor
64^2 runs and of the 4Q 128^2 run. On the held-out last fifth of each run's
time, on the faces the late rung answered, its pressures are about twice as
close as production's seed (median |d ln p| 0.023 against 0.047). It picks
the steering right 95% of the time and the rotation 84% (production's
suggestion: 71%).

The features are planar_data.rows_to_arrays's, operation for operation:
rotate about the normal so the LEFT tangential field lies along +y (the right
one if the left is zero, then reflect y if it points down); scale rho and P by
1/rho_L and B by 1/sqrt(rho_L); 13 numbers plus gamma. The outputs are mapped
back into the late rung's frame (planar5_b.to_planar) and physical units.
"""
from __future__ import annotations

import numpy as np

TINY = 1e-12
_CACHE = {}


def _load(path):
    """(network, x_mean, x_std, y_mean, y_std) of a planar_train.py
    checkpoint, once per process."""
    if path not in _CACHE:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        layers, w = [], int(ck["n_in"])
        for h in ck["hidden"]:
            layers += [torch.nn.Linear(w, int(h)), torch.nn.SiLU()]
            w = int(h)
        layers.append(torch.nn.Linear(w, 8))
        net = torch.nn.Sequential(*layers)
        # planar_train.PlanarMLP keeps the layers as `.net`
        net.load_state_dict({k[len("net."):]: v for k, v in ck["state_dict"].items()})
        net.eval()
        f32 = lambda a: np.asarray(a, dtype=np.float32)
        _CACHE[path] = (net, f32(ck["x_mean"]), f32(ck["x_std"]), f32(ck["y_mean"]), f32(ck["y_std"]))
    return _CACHE[path]


def features(left, right, Bn, gamma):
    """The lanes' features and canonical frame: ``(X (n, 14), ang, flip,
    rho_L)``. ``left``/``right`` are the seven solver-frame components (rho,
    total pressure, vx, vy, vz, By, Bz), ``Bn`` the normal field."""
    UL = np.stack([np.asarray(c, dtype=float) for c in left] + [np.asarray(Bn, dtype=float)], axis=1)
    UR = np.stack([np.asarray(c, dtype=float) for c in right] + [np.asarray(Bn, dtype=float)], axis=1)
    ByL, BzL = UL[:, 5], UL[:, 6]
    ByR, BzR = UR[:, 5], UR[:, 6]
    left_zero = np.hypot(ByL, BzL) <= TINY * np.maximum(1.0, np.abs(UL[:, 7]))
    ang = np.where(left_zero, np.arctan2(BzR, ByR), np.arctan2(BzL, ByL))
    ca, sa = np.cos(-ang), np.sin(-ang)

    def rot(X):
        Y = X.copy()
        Y[:, 3] = ca * X[:, 3] - sa * X[:, 4]
        Y[:, 4] = sa * X[:, 3] + ca * X[:, 4]
        Y[:, 5] = ca * X[:, 5] - sa * X[:, 6]
        Y[:, 6] = sa * X[:, 5] + ca * X[:, 6]
        return Y

    L, R = rot(UL), rot(UR)
    flip = left_zero & (R[:, 5] < 0.0)
    for X in (L, R):
        X[flip, 3] *= -1.0
        X[flip, 5] *= -1.0
    lam = 1.0 / L[:, 0]
    sq = np.sqrt(lam)
    Bn_s = UL[:, 7] * sq
    ByL_s, ByR_s = L[:, 5] * sq, R[:, 5] * sq
    with np.errstate(divide="ignore", invalid="ignore"):
        X = np.stack([np.log(L[:, 1] * lam), L[:, 2], L[:, 3], ByL_s, np.arcsinh(ByL_s / 1e-3),
                      np.log(R[:, 0] * lam), np.log(R[:, 1] * lam), R[:, 2], R[:, 3], ByR_s,
                      np.arcsinh(ByR_s / 1e-3), Bn_s, np.arcsinh(Bn_s / 1e-3),
                      np.full(len(Bn_s), float(gamma))], axis=1)
    return X, ang, flip, L[:, 0]


def predict(path, left, right, Bn, gamma, alpha):
    """The network's planar seed of the lanes (module doc). ``alpha`` is
    planar5_b.to_planar's rotation of the same lanes, the frame the late rung
    solves in. Returns ``dict(u_S, u_P, rot, pressure, ok)``; ``ok`` is False
    where a feature or an output is not finite (the caller falls back)."""
    import torch
    net, xm, xs, ym, ys = _load(path)
    X, ang, flip, rhoL = features(left, right, Bn, gamma)
    ok = np.isfinite(X).all(axis=1)
    Xn = (np.where(ok[:, None], X, 0.0).astype(np.float32) - xm) / xs
    with torch.no_grad():
        out = net(torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32))).numpy().astype(float)
    ok &= np.isfinite(out).all(axis=1)
    reg = out[:, :4] * ys.astype(float) + ym.astype(float)
    with np.errstate(all="ignore"):
        lnr = np.log(rhoL)
        lnpLF, lnpRF, lnPCD = reg[:, 0] + lnr, reg[:, 1] + lnr, reg[:, 3] + lnr
        by = reg[:, 2] * np.sqrt(rhoL)                 # canonical frame, physical units
        by = np.where(flip, -by, by)                   # undo the reflection
        by = by * np.cos(ang + np.asarray(alpha, dtype=float))   # into to_planar's frame
    return dict(u_S=[lnpLF, by, by.copy(), lnpRF],
                u_P=[lnpLF.copy(), lnPCD, lnPCD.copy(), lnpRF.copy()],
                rot=(out[:, 4:6] > 0.0).T.copy(),
                pressure=(out[:, 6] + out[:, 7]) > 0.0,
                ok=ok)
