"""
Spacetime metric in 3+1 decomposition.

Provides lapse, shift, spatial metric and index-raising/lowering operations.
Default is flat Minkowski: g = I, beta = 0, alpha = 1.
"""

import torch


class metric:
    def __init__(self, g, beta, alp):
        """
        Args:
            g    : (3,3) spatial metric tensor
            beta : (3,)  shift vector
            alp  : (1,)  lapse function
        """
        assert g.shape[0] == g.shape[1] == 3, "Spatial metric must be 3x3"
        self.sqrtg = torch.sqrt(torch.linalg.det(g))
        self.invg  = torch.linalg.inv(g)
        self.g     = g
        self.beta  = beta
        self.alp   = alp

    def raise_index(self, covec):
        """Raise the index of a covariant vector or batch: v^i = g^{ij} v_j."""
        if covec.dim() == 1:
            return torch.matmul(self.invg, covec)
        else:
            return torch.matmul(self.invg, covec.T).T

    def lower_index(self, vec):
        """Lower the index of a contravariant vector or batch: v_i = g_{ij} v^j."""
        if vec.dim() == 1:
            return torch.matmul(self.g, vec)
        else:
            return torch.matmul(self.g, vec.T).T

    def square_norm_upper(self, vec):
        """Square norm of a contravariant vector or batch: g_{ij} v^i v^j."""
        if vec.dim() == 1:
            return torch.matmul(vec, torch.matmul(self.g, vec))
        else:
            return torch.einsum('bi,ij,bj->b', vec, self.g, vec)

    def square_norm_lower(self, covec):
        """Square norm of a covariant vector or batch: g^{ij} v_i v_j."""
        if covec.dim() == 1:
            return torch.matmul(covec, torch.matmul(self.invg, covec))
        else:
            return torch.einsum('bi,ij,bj->b', covec, self.invg, covec)
