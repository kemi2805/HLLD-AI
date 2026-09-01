#!/usr/bin/env python3
"""
Quick test script to verify boundary condition fix
"""

import pytest

# SKIPPED (pre-existing, never ran): these were written against a data layout
# the solver does not use.  They build (1, ncells) 2-D tensors and index
# U["rho"][0, :2], but Grid1D stores flat (ntotal,) = (ncells + 2*ng,) tensors,
# and they call grid.apply_constant_bc(U) with one argument where the
# signature is apply_constant_bc(U, U_old).  Repairing them means rewriting
# against the real 1-D API, which Phase 2 replaces with Grid2D anyway — so
# they are parked here rather than deleted, to be rewritten as Grid2D BC
# tests (see tests/test_grid2d.py).
pytest.skip("BC tests target a stale array layout; superseded by Grid2D work",
            allow_module_level=True)



import torch
from src.physics.grid import Grid1D


def test_constant_bc_fix():
    """Test that the constant boundary condition fix works correctly."""
    print("Testing constant boundary condition fix...")

    # Create a simple grid with 5 physical cells and 2 ghost cells
    grid = Grid1D(xmin=0.0, xmax=1.0, ncells=5, ng=2)

    # Create test data with some values
    U = {
        "rho": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "px": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
    }

    print(f"Original data shape: {U['rho'].shape}")
    print(f"Original rho values: {U['rho'][0]}")
    print(f"Original px values: {U['px'][0]}")

    # Print grid info
    print(f"Grid ncells: {grid.ncells}")
    print(f"Grid ng: {grid.ng}")
    print(f"Grid ntotal: {grid.ntotal}")
    print(f"Physical region slice: {grid.phys}")

    # Store original values
    original_rho = U["rho"].clone()
    original_px = U["px"].clone()

    # Apply constant boundary conditions
    grid.apply_constant_bc(U)

    print(f"After applying constant BC:")
    print(f"rho values: {U['rho'][0]}")
    print(f"px values: {U['px'][0]}")

    # Check that ghost cells are properly filled
    # Left ghost cells (indices 0, 1) should equal first physical cell (index 2)
    left_ghost_rho = U["rho"][0, :2]
    left_ghost_px = U["px"][0, :2]
    first_physical_rho = original_rho[0, 2:3]
    first_physical_px = original_px[0, 2:3]

    print(f"Left ghost rho: {left_ghost_rho}")
    print(f"First physical rho: {first_physical_rho}")
    print(f"Left ghost px: {left_ghost_px}")
    print(f"First physical px: {first_physical_px}")

    # Verify the boundary conditions are applied correctly
    assert torch.equal(left_ghost_rho, first_physical_rho.expand_as(left_ghost_rho)), (
        "Left ghost cells not filled with first physical cell value for rho"
    )
    assert torch.equal(left_ghost_px, first_physical_px.expand_as(left_ghost_px)), (
        "Left ghost cells not filled with first physical cell value for px"
    )

    # Right ghost cells (indices 7, 8) should equal last physical cell (index 6)
    right_ghost_rho = U["rho"][0, -2:]
    right_ghost_px = U["px"][0, -2:]
    last_physical_rho = original_rho[0, -3:-2]
    last_physical_px = original_px[0, -3:-2]

    print(f"Right ghost rho: {right_ghost_rho}")
    print(f"Last physical rho: {last_physical_rho}")
    print(f"Right ghost px: {right_ghost_px}")
    print(f"Last physical px: {last_physical_px}")

    assert torch.equal(right_ghost_rho, last_physical_rho.expand_as(right_ghost_rho)), (
        "Right ghost cells not filled with last physical cell value for rho"
    )
    assert torch.equal(right_ghost_px, last_physical_px.expand_as(right_ghost_px)), (
        "Right ghost cells not filled with last physical cell value for px"
    )

    print("✓ Constant boundary condition fix verified successfully!")
    return True


if __name__ == "__main__":
    test_constant_bc_fix()
