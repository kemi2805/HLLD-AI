"""
1D uniform grid with ghost cells.

Ghost cell layout (ng=2 default):
  [ ghost | ghost | cell_0 | cell_1 | ... | cell_{N-1} | ghost | ghost ]
   idx 0     idx 1   idx 2   idx 3          idx N+1      idx N+2  idx N+3

Physical cells:  indices  ng  ..  ng+N-1
Left boundary :  indices   0  ..  ng-1
Right boundary:  indices  ng+N  ..  ng+N+ng-1
"""

import torch


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
        print("N = ", N)
        for k, v in U.items():
            v_old = U_old[k]
            # left ghosts  ← initial left ghost values
            v[:ng] = v_old[:ng]
            # right ghosts ← initial right ghost values
            v[ng + N :] = v_old[ng + N :]
        return U
