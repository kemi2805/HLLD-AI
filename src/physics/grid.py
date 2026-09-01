"""
Uniform grids with ghost cells (1D and 2D).

Ghost cell layout (ng=2 default):
  [ ghost | ghost | cell_0 | cell_1 | ... | cell_{N-1} | ghost | ghost ]
   idx 0     idx 1   idx 2   idx 3          idx N+1      idx N+2  idx N+3

Physical cells:  indices  ng  ..  ng+N-1
Left boundary :  indices   0  ..  ng-1
Right boundary:  indices  ng+N  ..  ng+N+ng-1
"""

import torch


def _sl(axis: int, s, ndim: int = 2) -> tuple:
    """Index tuple selecting ``s`` along ``axis`` and everything else."""
    out = [slice(None)] * ndim
    out[axis] = s
    return tuple(out)


class Grid1D:
    def __init__(
        self, xmin: float, xmax: float, ncells: int, ng: int = 2, device: str = "cpu"
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ncells = ncells  # number of physical cells
        self.ng = ng  # number of ghost cells on each side
        self.device = device

        self.dx = (xmax - xmin) / ncells
        self.ntotal = ncells + 2 * ng  # total cells including ghosts

        # Cell-centre coordinates (including ghosts)
        i = torch.arange(self.ntotal, dtype=torch.float64, device=device)
        self.x = xmin + (i - ng + 0.5) * self.dx  # shape (ntotal,)

        # Slice objects for physical region
        self.phys = slice(ng, ng + ncells)
        # Interface indices: flux[j] lives between cell j-1 and j (physical)
        # We compute fluxes at ng..ng+ncells interfaces (ncells+1 total)

    @property
    def x_phys(self) -> torch.Tensor:
        return self.x[self.phys]

    def apply_outflow_bc(self, U: dict) -> dict:
        """
        Outflow (zero-gradient) boundary conditions.
        Copies the outermost physical cell into all ghost cells.
        Works in-place and returns U.
        """
        ng = self.ng
        N = self.ncells
        for k, v in U.items():
            # left ghosts  ← first physical cell
            v[:ng] = v[ng].unsqueeze(0).expand(ng, *v.shape[1:])
            # right ghosts ← last physical cell
            v[ng + N :] = v[ng + N - 1].unsqueeze(0).expand(ng, *v.shape[1:])
        return U

    def apply_periodic_bc(self, U: dict) -> dict:
        """Periodic boundary conditions."""
        ng = self.ng
        N = self.ncells
        for k, v in U.items():
            v[:ng] = v[N : N + ng]
            v[N + ng :] = v[ng : 2 * ng]
        return U

    def apply_constant_bc(self, U: dict, U_old: dict) -> dict:
        """
        Constant boundary conditions.
        Holds ghost cells fixed at their initial/reference values from U_old.
        Works in-place and returns U.
        """
        ng = self.ng
        N = self.ncells
        for k, v in U.items():
            v_old = U_old[k]
            # left ghosts  ← initial left ghost values
            v[:ng] = v_old[:ng]
            # right ghosts ← initial right ghost values
            v[ng + N :] = v_old[ng + N :]
        return U


# ── 2D grid ──────────────────────────────────────────────────────────────────


def _bc_cc_axis(v: torch.Tensor, axis: int, n: int, ng: int, kind: str,
                v_ref: torch.Tensor | None = None) -> None:
    """Fill ghost entries of a CELL-CENTRED array along one axis, in place.

    Physical entries occupy ``ng .. ng+n-1`` along ``axis``.
    """
    if kind == "outflow":
        first = v[_sl(axis, slice(ng, ng + 1))]
        last = v[_sl(axis, slice(ng + n - 1, ng + n))]
        v[_sl(axis, slice(None, ng))] = first
        v[_sl(axis, slice(ng + n, None))] = last
    elif kind == "periodic":
        v[_sl(axis, slice(None, ng))] = v[_sl(axis, slice(n, n + ng))]
        v[_sl(axis, slice(n + ng, None))] = v[_sl(axis, slice(ng, 2 * ng))]
    elif kind == "constant":
        if v_ref is None:
            raise ValueError("constant BC needs a reference array")
        v[_sl(axis, slice(None, ng))] = v_ref[_sl(axis, slice(None, ng))]
        v[_sl(axis, slice(ng + n, None))] = v_ref[_sl(axis, slice(ng + n, None))]
    else:
        raise ValueError(f"unknown boundary condition {kind!r}")


def _bc_face_normal_axis(v: torch.Tensor, axis: int, n: int, ng: int,
                         kind: str) -> None:
    """Fill ghost entries of a FACE-STAGGERED array along its OWN normal axis.

    The array has ``n + 1`` physical faces at indices ``ng .. ng+n`` — note
    the inclusive upper end, one more than for cell-centred data, which is
    the usual source of off-by-one errors here.

    For a periodic axis the faces at ``ng`` and ``ng+n`` are the SAME physical
    face.  They are kept bit-identical by assigning the alias explicitly.
    Mathematically that is a no-op when the EMFs are periodic-consistent, but
    it is not a no-op in floating point, and skipping it is how div(B) drifts
    at the seam.
    """
    if kind == "outflow":
        first = v[_sl(axis, slice(ng, ng + 1))]
        last = v[_sl(axis, slice(ng + n, ng + n + 1))]
        v[_sl(axis, slice(None, ng))] = first
        v[_sl(axis, slice(ng + n + 1, None))] = last
    elif kind == "periodic":
        # collapse the duplicated seam face onto the single stored value
        v[_sl(axis, slice(ng + n, ng + n + 1))] = v[_sl(axis, slice(ng, ng + 1))]
        v[_sl(axis, slice(None, ng))] = v[_sl(axis, slice(n, n + ng))]
        v[_sl(axis, slice(ng + n + 1, None))] = v[_sl(axis, slice(ng + 1, ng + 1 + ng))]
    elif kind == "constant":
        pass  # ghost faces stay at their initial values
    else:
        raise ValueError(f"unknown boundary condition {kind!r}")


class Grid2D:
    """Uniform 2D Cartesian grid with ghost cells and staggered B storage.

    Cell-centred arrays have shape ``(nxt, nyt)`` with
    ``nxt = nx + 2*ng``.  Physical cells are ``[ng:ng+nx, ng:ng+ny]``.

    Face-staggered magnetic field:

        Bxf : (nxt + 1, nyt)   Bxf[i, j] is the LOWER-x face of cell (i, j)
        Byf : (nxt, nyt + 1)   Byf[i, j] is the LOWER-y face of cell (i, j)

    with corner EMFs on ``(nxt + 1, nyt + 1)``.  This is the GRACE
    convention: index ``i`` refers to the face at ``x_{i-1/2}``.

    Ghost width
    -----------
    ``ng = 2`` is exactly sufficient for PLM + constrained transport.  PLM
    slopes exist for cells ``1 .. nxt-2``, so two-sided faces are exactly the
    ``nx+1`` physical x-faces — no spare.  The CT "one face wider" requirement
    is in the TRANSVERSE index, where no reconstruction happens, so it is
    covered by the ghost rows for free.  PPM or WENO would need ``ng = 3``.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        nx: int,
        ymin: float,
        ymax: float,
        ny: int,
        ng: int = 2,
        device: str = "cpu",
    ):
        self.xmin, self.xmax, self.nx = xmin, xmax, nx
        self.ymin, self.ymax, self.ny = ymin, ymax, ny
        self.ng = ng
        self.device = device

        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny
        self.nxt = nx + 2 * ng
        self.nyt = ny + 2 * ng

        i = torch.arange(self.nxt, dtype=torch.float64, device=device)
        j = torch.arange(self.nyt, dtype=torch.float64, device=device)
        self.x = xmin + (i - ng + 0.5) * self.dx          # cell centres
        self.y = ymin + (j - ng + 0.5) * self.dy
        # face coordinates: xf[i] is the lower-x face of cell i
        self.xf = xmin + (torch.arange(self.nxt + 1, dtype=torch.float64,
                                       device=device) - ng) * self.dx
        self.yf = ymin + (torch.arange(self.nyt + 1, dtype=torch.float64,
                                       device=device) - ng) * self.dy

        self.X = self.x.unsqueeze(1).expand(self.nxt, self.nyt)
        self.Y = self.y.unsqueeze(0).expand(self.nxt, self.nyt)

        self.phys = (slice(ng, ng + nx), slice(ng, ng + ny))
        self.phys_xf = (slice(ng, ng + nx + 1), slice(ng, ng + ny))
        self.phys_yf = (slice(ng, ng + nx), slice(ng, ng + ny + 1))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nxt, self.nyt)

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.nxt, self.nyt, dtype=torch.float64,
                           device=self.device)

    def zeros_xf(self) -> torch.Tensor:
        return torch.zeros(self.nxt + 1, self.nyt, dtype=torch.float64,
                           device=self.device)

    def zeros_yf(self) -> torch.Tensor:
        return torch.zeros(self.nxt, self.nyt + 1, dtype=torch.float64,
                           device=self.device)

    def zeros_corner(self) -> torch.Tensor:
        return torch.zeros(self.nxt + 1, self.nyt + 1, dtype=torch.float64,
                           device=self.device)

    # ── boundary conditions ──────────────────────────────────────────────

    def apply_bc_cc(self, U: dict, bc_x: str = "outflow", bc_y: str = "outflow",
                    U_ref: dict | None = None) -> dict:
        """Apply BCs to a dict of cell-centred ``(nxt, nyt)`` tensors, in place.

        x is filled first, then y, so the corner ghost blocks inherit the
        y-treatment of already-filled x-ghosts (the usual convention).
        """
        for k, v in U.items():
            ref = None if U_ref is None else U_ref[k]
            _bc_cc_axis(v, 0, self.nx, self.ng, bc_x, ref)
            _bc_cc_axis(v, 1, self.ny, self.ng, bc_y, ref)
        return U

    def apply_bc_face(self, Bxf: torch.Tensor, Byf: torch.Tensor,
                      bc_x: str = "outflow", bc_y: str = "outflow"
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply BCs to the staggered field, in place.

        Each component uses the face rule along its own normal axis and the
        ordinary cell-centred rule along the transverse axis.
        """
        _bc_face_normal_axis(Bxf, 0, self.nx, self.ng, bc_x)
        _bc_cc_axis(Bxf, 1, self.ny, self.ng, bc_y)

        _bc_cc_axis(Byf, 0, self.nx, self.ng, bc_x)
        _bc_face_normal_axis(Byf, 1, self.ny, self.ng, bc_y)
        return Bxf, Byf

    def div_b(self, Bxf: torch.Tensor, Byf: torch.Tensor) -> torch.Tensor:
        """Cell-centred div(B) from the staggered field; shape ``(nxt, nyt)``.

        Exact to machine precision for a field built from a vector potential
        and preserved by the CT update.
        """
        out = torch.zeros_like(self.X)
        out[:, :] = (Bxf[1:, :] - Bxf[:-1, :]) / self.dx \
            + (Byf[:, 1:] - Byf[:, :-1]) / self.dy
        return out
