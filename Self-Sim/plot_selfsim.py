import matplotlib.pyplot as plt
import numpy as np

snaps = [
    "./ken_output_long_ai/snap_000100.npz",
    "./ken_output_long_ai/snap_000200.npz",
    "./ken_output_long_ai/snap_000400.npz",
]

fig, axes = plt.subplots(3, 1, figsize=(8, 10))

x_interface = 0.5  # or wherever your discontinuity starts

for fname in snaps:
    d = np.load(fname)
    t = float(d["t"])
    x = d["x"]
    xi = (x - x_interface) / t  # ← centred on the interface

    axes[0].plot(xi, d["prim_rho"], label=f"t={t:.3f}")
    axes[1].plot(xi, d["prim_vx"], label=f"t={t:.3f}")
    axes[2].plot(xi, d["prim_p"], label=f"t={t:.3f}")

# Also set symmetric x-limits so both left/right waves are visible
for ax in axes:
    ax.set_xlim(-5, 5)
    ax.set_xlabel("ξ = (x - x₀) / t")


for ax, ylabel in zip(axes, ["rho", "vx", "p"]):
    ax.set_xlabel("ξ = x/t")
    ax.set_ylabel(ylabel)
    ax.legend()

plt.tight_layout()
plt.savefig("selfsimilar-ai.png", dpi=150)
