"""1D Special-Relativistic MHD (SR-MHD) codebase with HLLD Riemann solver.

Physics solver
--------------
    from src.physics import hybrid_eos, Grid1D, hlld_flux, conservative_to_primitive

ML pipeline
-----------
    from src.data     import HLLDDataset, get_dataloaders
    from src.models   import PressureNet
    from src.training import train, evaluate

Run a shocktube simulation
--------------------------
    python -m src.physics.driver          # reads configs/shocktube.yaml by default
    python src/physics/run_shocktube.py   # legacy script entry-point

Generate training data
----------------------
    python -m src.data.generate --config configs/default.yaml

Train the pressure-prediction network
--------------------------------------
    python -m src.training.train --config configs/default.yaml
"""
