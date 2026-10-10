import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = r"d:\sedimentation_dynamics"
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
data_dir = os.path.join(mb_custom, "data")
plots_dir = os.path.join(mb_custom, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

# Load JSON summary data
summary_file = os.path.join(data_dir, "trajectories_summary.json")
with open(summary_file, "r") as f:
    data = json.load(f)

# Extract single trajectories
times = np.array(data["sphere"]["single"]["0.0"]["times"])
dt = times[1] - times[0]
T = times[-1]

single_trajs = {
    "Sphere (N=42)": {
        "traj": np.array(data["sphere"]["single"]["0.0"]["traj_b1"]),
        "color": "#1f77b4",
        "style": "-",
        "F": 1000.0
    },
    "Disc (N=37, Broadside)": {
        "traj": np.array(data["disc"]["single"]["0.0"]["traj_b1"]),
        "color": "#ff7f0e",
        "style": "-",
        "F": 1000.0
    },
    "Cylinder (N=86, Horiz)": {
        "traj": np.array(data["cylinder"]["single"]["0.0"]["traj_b1"]),
        "color": "#2ca02c",
        "style": "-",
        "F": 1000.0
    },
    "Ellipsoid (N=42, θ=0°)": {
        "traj": np.array(data["ellipsoid"]["ang_0"]["0.0"]["traj_b1"]),
        "color": "#d62728",
        "style": "-",
        "F": 1000.0
    },
    "Ellipsoid (N=42, θ=45°)": {
        "traj": np.array(data["ellipsoid"]["ang_45"]["0.0"]["traj_b1"]),
        "color": "#9467bd",
        "style": "--",
        "F": 1000.0
    },
    "Ellipsoid (N=42, θ=90°)": {
        "traj": np.array(data["ellipsoid"]["ang_90"]["0.0"]["traj_b1"]),
        "color": "#8c564b",
        "style": "-.",
        "F": 1000.0
    },
    "Boomerang (N=15, F=300)": {
        "traj": np.array(data["boomerang"]["single"]["0.0"]["traj_b1"]),
        "color": "#e377c2",
        "style": ":",
        "F": 300.0
    }
}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# =========================================================================
# Panel 1: Vertical Displacement z(t) vs Time
# =========================================================================
ax1 = axes[0, 0]
for name, info in single_trajs.items():
    z = info["traj"][:, 2]
    vz = abs(z[-1] - z[0]) / T
    ax1.plot(times, z, info["style"], color=info["color"], linewidth=2.2,
             label=f"{name} (Vz = {vz:.1f})")

ax1.set_title("Single Body Vertical Displacement z(t) vs Time", fontsize=12, fontweight="bold")
ax1.set_xlabel("Time t (s)", fontsize=11)
ax1.set_ylabel("Vertical Position z", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=9, loc="lower left")

# =========================================================================
# Panel 2: Spatial (x vs z) Trajectory Flight Path
# =========================================================================
ax2 = axes[0, 1]
for name, info in single_trajs.items():
    x = info["traj"][:, 0]
    z = info["traj"][:, 2]
    ax2.plot(x, z, info["style"], color=info["color"], linewidth=2.2, label=name)

ax2.set_title("Spatial Flight Paths (x vs z Plane)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Lateral Position x", fontsize=11)
ax2.set_ylabel("Vertical Position z", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=9, loc="lower left")
ax2.annotate("Lateral Glide Drift\n(θ = 45° Pitch Tilt)", xy=(5.5, -38), xytext=(2.0, -20),
             arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.5),
             fontsize=9, fontweight="bold", color="#9467bd")

# =========================================================================
# Panel 3: Terminal Settling Velocity Bar Chart
# =========================================================================
ax3 = axes[1, 0]
names = list(single_trajs.keys())
vels = [abs(single_trajs[n]["traj"][-1, 2] - single_trajs[n]["traj"][0, 2]) / T for n in names]
cols = [single_trajs[n]["color"] for n in names]

bars = ax3.barh(names, vels, color=cols, alpha=0.85, edgecolor="black", height=0.6)
for bar, v in zip(bars, vels):
    ax3.text(v + 1.5, bar.get_y() + bar.get_height()/2.0, f"{v:.2f}",
             va="center", ha="left", fontsize=10, fontweight="bold")

ax3.set_xlim(0, max(vels) * 1.15)
ax3.set_title("Terminal Settling Velocity Comparison (V0)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Terminal Velocity |V_z|", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.6, axis="x")
ax3.invert_yaxis()

# =========================================================================
# Panel 4: Geometry of the Disc Discretization (N=37)
# =========================================================================
ax4 = axes[1, 1]

# Load disc vertex coordinates
disc_vertex_file = os.path.join(mb_custom, "structures", "disc_N_37.vertex")
with open(disc_vertex_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]
N_disc = int(lines[0])
disc_pts = np.array([[float(p) for p in l.split()] for l in lines[1:]])

blob_rad = 0.165
# Plot hydrodynamic spheres as circular patches
for pt in disc_pts:
    circle = patches.Circle((pt[0], pt[1]), blob_rad, color="#1f77b4", alpha=0.35, edgecolor="#0d47a1", linewidth=1.2)
    ax4.add_patch(circle)
ax4.plot(disc_pts[:, 0], disc_pts[:, 1], "ko", markersize=3.5, label="Blob Centers (N=37)")

# Draw ring guides
for r, ring_label in zip([0.333, 0.667, 1.0], ["Ring 1 (6 blobs)", "Ring 2 (12 blobs)", "Ring 3 (18 blobs)"]):
    ring_circ = patches.Circle((0, 0), r, fill=False, linestyle="--", color="crimson", alpha=0.7, linewidth=1.0)
    ax4.add_patch(ring_circ)

# Outer disc boundary
outer_circ = patches.Circle((0, 0), 1.0, fill=False, linestyle="-", color="black", linewidth=2.0, label="Disc Boundary R=1.0")
ax4.add_patch(outer_circ)

ax4.set_aspect("equal")
ax4.set_xlim(-1.35, 1.35)
ax4.set_ylim(-1.35, 1.35)
ax4.set_title("Disc N=37 Concentric Discretization & Hydrodynamic Overlap", fontsize=12, fontweight="bold")
ax4.set_xlabel("x Coordinate", fontsize=11)
ax4.set_ylabel("y Coordinate", fontsize=11)
ax4.grid(True, linestyle="--", alpha=0.5)

info_txt = (
    "Discretization Structure:\n"
    "• Center: 1 blob\n"
    "• Ring 1 (r = R/3): 6 blobs\n"
    "• Ring 2 (r = 2R/3): 12 blobs\n"
    "• Ring 3 (r = R): 18 blobs\n"
    "Total N = 1 + 6 + 12 + 18 = 37\n"
    "• Spacing s ≈ 0.34 R\n"
    "• Blob radius a = 0.165 R\n"
    "• 2a ≈ s (Zero-leakage threshold)"
)
ax4.text(1.4, 0.0, info_txt, fontsize=9.5, va="center", ha="left",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9))

plt.tight_layout()

# Save
p1 = os.path.join(plots_dir, "single_body_trajectories_comparison.png")
p2 = os.path.join(thesis_fig_dir, "single_body_trajectories_comparison.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Generated single body comparison plot: {p1} and {p2}")
