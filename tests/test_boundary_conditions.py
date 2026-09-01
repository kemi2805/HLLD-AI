"""
Test boundary condition implementations.
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


def test_constant_boundary_condition():
    """Test that constant boundary conditions work correctly."""
    # Create a simple grid
    grid = Grid1D(xmin=0.0, xmax=1.0, ncells=5, ng=2)

    # Create some test data
    U = {
        "rho": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "px": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "py": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "pz": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "e": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype.float64),
    }

    # Store original values for comparison
    original_values = {k: v.clone() for k, v in U.items()}

    # Apply constant boundary conditions
    grid.apply_constant_bc(U)

    # Check that ghost cells were filled with boundary values
    # Left ghost cells should equal first physical cell
    for k, v in U.items():
        assert torch.equal(v[0], original_values[k][0]), (
            f"Left ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[1], original_values[k][0]), (
            f"Left ghost cell not set correctly for {k}"
        )
        # Right ghost cells should equal last physical cell
        assert torch.equal(v[-1], original_values[k][-1]), (
            f"Right ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[-2], original_values[k][-1]), (
            f"Right ghost cell not set correctly for {k}"
        )


def test_outflow_boundary_condition():
    """Test that outflow boundary conditions work correctly."""
    # Create a simple grid
    grid = Grid1D(xmin=0.0, xmax=1.0, ncells=5, ng=2)

    # Create some test data
    U = {
        "rho": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "px": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "py": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "pz": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "e": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
    }

    # Store original values for comparison
    original_values = {k: v.clone() for k, v in U.items()}

    # Apply outflow boundary conditions
    grid.apply_outflow_bc(U)

    # Check that ghost cells were filled with boundary values
    # Left ghost cells should equal first physical cell
    for k, v in U.items():
        assert torch.equal(v[0], original_values[k][0]), (
            f"Left ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[1], original_values[k][0]), (
            f"Left ghost cell not set correctly for {k}"
        )
        # Right ghost cells should equal last physical cell
        assert torch.equal(v[-1], original_values[k][-1]), (
            f"Right ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[-2], original_values[k][-1]), (
            f"Right ghost cell not set correctly for {k}"
        )


def test_periodic_boundary_condition():
    """Test that periodic boundary conditions work correctly."""
    # Create a simple grid
    grid = Grid1D(xmin=0.0, xmax=1.0, ncells=5, ng=2)

    # Create some test data
    U = {
        "rho": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "px": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "py": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "pz": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
        "e": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64),
    }

    # Store original values for comparison
    original_values = {k: v.clone() for k, v in U.items()}

    # Apply periodic boundary conditions
    grid.apply_periodic_bc(U)

    # Check that ghost cells were filled with periodic values
    # Left ghost cells should equal last physical cells
    for k, v in U.items():
        assert torch.equal(v[0], original_values[k][-2]), (
            f"Left ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[1], original_values[k][-1]), (
            f"Left ghost cell not set correctly for {k}"
        )
        # Right ghost cells should equal first physical cells
        assert torch.equal(v[-1], original_values[k][1]), (
            f"Right ghost cell not set correctly for {k}"
        )
        assert torch.equal(v[-2], original_values[k][0]), (
            f"Right ghost cell not set correctly for {k}"
        )


if __name__ == "__main__":
    test_constant_boundary_condition()
    test_outflow_boundary_condition()
    test_periodic_boundary_condition()
    print("All boundary condition tests passed!")
