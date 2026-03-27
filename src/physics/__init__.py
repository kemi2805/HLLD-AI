"""Physics modules for the 1D SR-MHD HLLD solver.

Core building blocks
--------------------
hybrid_eos                — equation of state (cold polytrope + thermal ideal gas)
metric                    — 3+1 spacetime metric (index raising/lowering)
Grid1D                    — 1D uniform grid with ghost cells
hlld_flux                 — HLLD Riemann solver (main interface)
conservative_to_primitive — Kastaun (2021) vectorised C2P for SR-MHD
get_interface_states      — PLM reconstruction (MC / minmod limiters)
init_shocktube            — grid initialisation from left/right primitive states

Legacy (ML training pipeline)
------------------------------
physics_utils functions (W__z, rho__z, …) — pure-hydro C2P used by src.data.generate
"""

from .eos import hybrid_eos
from .metric import metric
from .grid import Grid1D
from .initial_data import (
    brio_wu, komissarov_1, komissarov_2, generic, init_shocktube,
)
from .reconstruction import reconstruct_plm, get_interface_states, minmod, mc_limiter
from .hlld import (
    hlld_flux,
    hlle_flux,
    HLLDComputation,
    compute_srmhd_fluxes,
    primitive_to_conserved,
    primitive_state,
    lorentz,
    compute_b2,
    compute_smallb,
    safe_secant_bisection,
)
from .c2p import conservative_to_primitive, KastaunC2P

# Legacy ML-era pure-hydro C2P utilities — kept for src.data.generate compatibility
from .physics_utils import (
    W__z, rho__z, eps__z, h__z, a__z,
    sanity_check, compute_primitives,
    conservative_to_primitive_exact,
    validate_conservative_variables,
)

__all__ = [
    # EOS & metric
    'hybrid_eos', 'metric',
    # Grid
    'Grid1D',
    # Initial data
    'brio_wu', 'komissarov_1', 'komissarov_2', 'generic', 'init_shocktube',
    # Reconstruction
    'reconstruct_plm', 'get_interface_states', 'minmod', 'mc_limiter',
    # Riemann solvers
    'hlld_flux', 'hlle_flux', 'HLLDComputation', 'compute_srmhd_fluxes',
    'primitive_to_conserved', 'primitive_state', 'lorentz',
    'compute_b2', 'compute_smallb', 'safe_secant_bisection',
    # Conservative-to-primitive
    'conservative_to_primitive', 'KastaunC2P',
    # Legacy ML-era utilities
    'W__z', 'rho__z', 'eps__z', 'h__z', 'a__z',
    'sanity_check', 'compute_primitives',
    'conservative_to_primitive_exact', 'validate_conservative_variables',
]
