"""1D and 2D special-relativistic MHD with HLLD, HLLE, HLLC and exact fluxes.

Physics solver
--------------
    from src.physics import hybrid_eos, Grid1D, hlld_flux, conservative_to_primitive

Run a simulation
----------------
    python scripts/shocktube_compare.py configs/Giacomazzo/Balsara1.yaml --solvers hlld
                                                # 1D: driver.run on any YAML config
    python scripts/run_2d.py --problem rotor    # 2D production driver

The exact flux (``--solver exact``) needs the rmhd package installed:
``pip install -e <rmhd_final checkout>``.  Its ML warm start lives there.
This repository's own p*-surrogate pipeline (src/data, src/models,
src/training, the ``hlld_ai`` solver) was deleted on 2026-09-17; it is in the
git history before that date (857cbfc is the last commit with it).  Its
untracked datasets and checkpoints (2.3 GB, never committed) are kept in
data/ARCHIVE_pstar_surrogate/, whose README says what each file is.
"""
