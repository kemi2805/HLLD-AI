"""
Piecewise Linear Method (PLM) reconstruction.

Given cell-averaged primitives Q[i] over the full domain (including ghosts),
returns interface states:
    QL[j]  — left  state at interface between cell j-1 and cell j   (right side of cell j-1)
    QR[j]  — right state at interface between cell j-1 and cell j   (left  side of cell j)

Indices run over the *physical* interfaces:  j = ng .. ng+N  (N+1 interfaces)

All operations are batched over the variable axis and vectorised with torch —
no Python loops over cells.
"""

import torch


# ── slope limiters ────────────────────────────────────────────────────────────

def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minmod limiter: returns the smaller-magnitude slope if same sign, else 0."""
    return torch.where(
        a * b > 0.0,
        torch.where(a.abs() < b.abs(), a, b),
        torch.zeros_like(a),
    )


def mc_limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Monotonized-Central (van Leer) limiter."""
    c = 0.5 * (a + b)
    return torch.where(
        a * b > 0.0,
        torch.sign(c) * torch.minimum(2.0 * a.abs(),
                        torch.minimum(2.0 * b.abs(), c.abs())),
        torch.zeros_like(a),
    )


# ── PLM reconstruction ────────────────────────────────────────────────────────

def reconstruct_plm(
    Q: torch.Tensor,
    limiter: str = "mc",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PLM reconstruction.

    Parameters
    ----------
    Q : (ntotal, nvars)  — cell-averaged primitives on full domain (with ghosts).
    limiter : "mc" | "minmod"

    Returns
    -------
    QL : (ntotal-1, nvars)  — left  state at each interface  (right edge of left cell)
    QR : (ntotal-1, nvars)  — right state at each interface  (left  edge of right cell)

    Interface j (0-based) sits between cell j and cell j+1 in the full array,
    so the caller slices QL/QR to the physical interfaces it needs.
    """
    lim = mc_limiter if limiter == "mc" else minmod

    # forward / backward differences
    dQf = Q[1:]   - Q[:-1]          # shape (ntotal-1, nvars)
    dQb = Q[:-1]  - Q[1:]           # = -dQf, but keep explicit for clarity

    # centred slopes at each cell centre:  slope[i] = lim( Q[i]-Q[i-1], Q[i+1]-Q[i] )
    # dQb[i] = Q[i] - Q[i+1]  →  we want Q[i] - Q[i-1] = dQf[i-1]
    # so slope[i] = lim( dQf[i-1],  dQf[i] )  for i in 1..ntotal-2
    slope = lim(dQf[:-1], dQf[1:])   # shape (ntotal-2, nvars)

    # Reconstruct at interfaces
    # Interface j sits between cell j and cell j+1 (0-based full indexing).
    # QL[j] = Q[j]   + 0.5 * slope[j-1]   (right edge of cell j,   j>=1)
    # QR[j] = Q[j+1] - 0.5 * slope[j]     (left  edge of cell j+1, j>=1)
    #
    # slope has indices 0..ntotal-3 corresponding to cell centres 1..ntotal-2.
    # Interface indices 1..ntotal-2 (ntotal-2 interfaces):
    QL = Q[1:-1] + 0.5 * slope        # shape (ntotal-2, nvars)
    QR = Q[2:]   - 0.5 * slope        # shape (ntotal-2, nvars)  (Q[j+1] = Q[2:])

    return QL, QR

def reconstruct_pcm(
    Q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCM reconstruction — piecewise constant, first-order."""
    QL = Q[:-1]   # right edge of left cell = cell average
    QR = Q[1:]    # left  edge of right cell = cell average
    return QL, QR


def get_interface_states(
    prims: dict[str, torch.Tensor],
    ng: int,
    ncells: int,
    limiter: str = "pcm",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Reconstruct all primitive variables and return left/right interface states
    for the *physical* interfaces (ncells+1 of them).

    Parameters
    ----------
    prims  : dict of (ntotal,) tensors — primitives on full domain incl. ghosts.
    ng     : number of ghost cells on each side.
    ncells : number of physical cells.

    Returns
    -------
    primL, primR : dicts of (ncells+1,) tensors — interface states.
    """
    keys = list(prims.keys())
    nvars = len(keys)
    ntotal = next(iter(prims.values())).shape[0]

    # Stack into a single (ntotal, nvars) tensor for vectorised reconstruction
    Q = torch.stack([prims[k] for k in keys], dim=1)  # (ntotal, nvars)

    if limiter == "pcm":
        QL_all, QR_all = reconstruct_pcm(Q)
    else:
        QL_all, QR_all = reconstruct_plm(Q, limiter=limiter)



    # QL_all[j] is the right edge of cell (j+1) in full indexing.
    # Physical interfaces run from j = ng-1 .. ng+ncells-1  in QL_all indexing
    # (because QL_all has shape ntotal-2, indexed 0..ntotal-3,
    #  and QL_all[j] corresponds to the right edge of full cell j+1).
    #
    # Full-cell interface between physical cell i and i+1 corresponds to
    # QL_all index  i + ng - 1  (for i = 0..ncells, giving ncells+1 interfaces).
    i0 = ng - 1
    i1 = i0 + ncells + 1   # exclusive

    QL_phys = QL_all[i0:i1]   # (ncells+1, nvars)
    QR_phys = QR_all[i0:i1]   # (ncells+1, nvars)

    primL = {k: QL_phys[:, j] for j, k in enumerate(keys)}
    primR = {k: QR_phys[:, j] for j, k in enumerate(keys)}

    return primL, primR