"""1D and 2D special-relativistic MHD with HLLD, HLLE, HLLC and exact fluxes.

Physics solver
--------------
    from src.physics import hybrid_eos, Grid1D, hlld_flux, conservative_to_primitive

Run a simulation
----------------
    python -m src.physics.driver          # 1D; pass a config path
    python scripts/run_2d.py --problem rotor    # 2D production driver

The exact flux (``--solver exact``) needs the rmhd package installed:
``pip install -e <rmhd_final checkout>``.  Its ML warm start lives there.
This repository's own p*-surrogate pipeline (src/data, src/models,
src/training, the ``hlld_ai`` solver) was deleted on 2026-09-17; it is in the
git history before that date.
"""
