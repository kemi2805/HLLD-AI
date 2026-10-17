"""
Training data generation — fully vectorized with PyTorch.

All N samples are generated in a single batched pass on the target device
(MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback).
No Python loop over samples.

Bx is shared between L and R states (divergence-free constraint).
Velocities are sampled via (W, theta, phi) to guarantee sub-luminal speeds.
Transverse B is sampled via (|Bt|, phi) for isotropic directions.

Feature vector (N, 15):
    [ rho_L, vx_L, vy_L, vz_L, p_L, By_L, Bz_L,
      rho_R, vx_R, vy_R, vz_R, p_R, By_R, Bz_R,
      Bx ]

Usage:
    python -m src.data.generate --config configs/default.yaml
"""

import argparse
import os

import numpy as np
import torch
import yaml

from src.physics.eos import hybrid_eos
from src.physics import ai_features as _ai
from src.physics.hlld import (
    Cons,
    compute_b2,
    compute_smallb,
    compute_srmhd_fluxes,
    energy_form,
    hlld_flux,
    lorentz,
)
from src.physics.metric import metric


def get_device_and_dtype(device_str: str):
    """
    Resolve device and dtype.
    MPS (Apple Silicon) does not support float64 — use float32 there.
    """
    if device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float32
        print("MPS not available, falling back to CPU.")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float64
        print("CUDA not available, falling back to CPU.")
    return torch.device("cpu"), torch.float64


def _uniform(lo: float, hi: float, n: int, device, dtype) -> torch.Tensor:
    """Sample n values uniformly in [lo, hi]."""
    return lo + (hi - lo) * torch.rand(n, device=device, dtype=dtype)


def sample_states(
    n: int,
    ranges: dict,
    Bx: torch.Tensor,
    eos: hybrid_eos,
    device,
    dtype,
    n_angles: int = 5,
) -> dict:
    """
    Sample n primitive states as batched tensors on device.

    Returns a dict with keys: rho, vx, vy, vz, p, eps, Bx, By, Bz
    Each value is a 1-D tensor of length n.
    """

    # Density: log-uniform
    lrho = _uniform(*ranges["lrho"], n, device, dtype)  # type:ignore
    rho = 10.0**lrho

    # Pressure: log_uniform
    lp = _uniform(*ranges["press"], n, device, dtype)  # type:ignore
    # p = 10.0 ** lp
    p = lp

    eps = eos.eps__press_rho(p, rho)

    # Velocity: sample W (Lorentz factor), then angles
    W = _uniform(*ranges["W"], n, device, dtype)  # type:ignore
    v_mag = torch.sqrt(1.0 - 1.0 / W**2)

    cos_theta = _uniform(-1.0, 1.0, n, device, dtype)  # uniform on sphere
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=0.0))
    phi_v0 = _uniform(0.0, 2.0 * torch.pi, n, device, dtype)

    # Transverse B: sample magnitude and angle
    phi_B0 = _uniform(0.0, 2.0 * torch.pi, n, device, dtype)
    Bt = _uniform(*ranges["Bt_mag"], n, device, dtype)  # type:ignore

    # 5 rotations per sample, stacked along dim 0
    steps = torch.arange(n_angles, device=device, dtype=dtype) * (
        2.0 * torch.pi / n_angles
    )

    vy_list, vz_list, By_list, Bz_list = [], [], [], []
    for k in range(n_angles):
        phi_v = (phi_v0 + steps[k]) % (2.0 * torch.pi)
        phi_B = phi_B0  # + steps[k]) % (2.0 * torch.pi)
        vy_list.append(v_mag * sin_theta * torch.cos(phi_v))
        vz_list.append(v_mag * sin_theta * torch.sin(phi_v))
        By_list.append(Bt * torch.cos(phi_B))
        Bz_list.append(Bt * torch.sin(phi_B))

    # each list: n_angles tensors of shape (n,) → cat to (n_angles*n,)
    rho = rho.repeat(n_angles)
    p = p.repeat(n_angles)
    eps = eps.repeat(n_angles)
    vx = (v_mag * cos_theta).repeat(n_angles)
    vy = torch.cat(vy_list)
    vz = torch.cat(vz_list)
    By = torch.cat(By_list)
    Bz = torch.cat(Bz_list)
    Bx = Bx.repeat(n_angles)

    return {
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "p": p,
        "eps": eps,
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
    }


def generate_dataset(
    n_samples: int,
    ranges: dict,
    eos_cfg: dict,
    device_str: str = "cpu",
    seed: int = 42,
    batch_size: int = 10_000,
    n_angles: int = 5,
):
    """
    Generate n_samples Riemann problems in batched passes to control memory.

    Returns:
        inputs  : (n_samples, 15) numpy array
        targets : (n_samples,)   numpy array  — total pressure p_tot*
    """
    device, dtype = get_device_and_dtype(device_str)
    torch.manual_seed(seed)
    print(f"Generating on {device} ({dtype}), batch_size={batch_size}")

    eos = hybrid_eos(**eos_cfg)
    met = metric(
        torch.eye(3, device=device, dtype=dtype),
        torch.zeros(3, device=device, dtype=dtype),
        torch.ones(1, device=device, dtype=dtype),
    )

    all_inputs = []
    all_targets = []

    n_done = 0
    batch_idx = 0
    while n_done < n_samples:
        n_batch = min(batch_size, n_samples - n_done)

        Bx = _uniform(*ranges["Bx"], n_batch, device, dtype)  # type:ignore
        state_L = sample_states(n_batch, ranges, Bx, eos, device, dtype, n_angles)
        state_R = sample_states(n_batch, ranges, Bx, eos, device, dtype, n_angles)
        Bx = state_L["Bx"]  # same for L and R, repeated for n_angles

        # inputs = torch.stack([
        #    state_L["rho"], state_L["vx"], state_L["vy"], state_L["vz"],
        #    state_L["p"],   state_L["By"], state_L["Bz"],
        #    state_R["rho"], state_R["vx"], state_R["vy"], state_R["vz"],
        #    state_R["p"],   state_R["By"], state_R["Bz"],
        #    Bx,
        # ], dim=1)

        _, _, p_star = hlld_flux(state_L, state_R, eos)

        uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(state_L, state_R, eos, 0)
        RL = {k: cmin * uL[k] - fL[k] for k in uL}
        RR = {k: cmax * uR[k] - fR[k] for k in uR}

        # –– Calculate new set of conserved variables. ——————————————————
        lfacL, _ = lorentz(state_L)
        lfacR, _ = lorentz(state_R)

        # ── Total-energy form + features (shared with the inference path) ────
        # This block used to be a third hand-written copy of the U_init /
        # F_init algebra and the 15-feature vector, all hardcoded to x.  Its
        # existence is why the feature vector could not be made
        # direction-generic: fixing hlld_ai_flux alone would have silently
        # desynchronised training from inference.  Both now call the same
        # code in src/physics/ai_features.py.
        U_init_L, F_init_L = energy_form(uL, fL)
        U_init_R, F_init_R = energy_form(uR, fR)

        RL_init, RR_init = _ai.r_vectors(U_init_L, F_init_L,
                                         U_init_R, F_init_R, cmin, cmax)
        inputs = _ai.build_pstar_features(state_L, state_R, uL, uR,
                                          RL_init, RR_init, cmin, cmax,
                                          idir=0)

        ## Swap L and R states to double the dataset size (symmetry of Riemann problem)
        # inputs_swapped = torch.cat([inputs[:, 6:12], inputs[:, 0:6], inputs[:, 12:15]], dim=1)
        # inputs  = torch.cat([inputs, inputs_swapped], dim=0)
        # p_star  = torch.cat([p_star, p_star],         dim=0)

        # Move to CPU immediately to free device memory
        all_inputs.append(inputs.cpu().numpy())
        all_targets.append(p_star.cpu().numpy())

        n_done += n_batch  # 2*n_angles * n_batch
        batch_idx += 1
        print(f"  batch {batch_idx}: {n_angles * n_done}/{n_samples}")

    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_path = args.output or cfg["data"]["raw_path"]

    inputs, targets = generate_dataset(
        n_samples=cfg["data"]["n_samples"],
        ranges=cfg["data"]["ranges"],
        eos_cfg=cfg["eos"],
        device_str=cfg["training"].get("device", "cpu"),
        seed=cfg["data"].get("seed", 42),
        batch_size=cfg["data"].get("batch_size", 10_000),
        n_angles=cfg["data"].get("n_angles", 5),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, inputs=inputs, targets=targets)
    print(f"Saved {len(inputs)} samples → {output_path}")


if __name__ == "__main__":
    main()
