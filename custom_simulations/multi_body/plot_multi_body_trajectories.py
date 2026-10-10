import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
data_dir = os.path.join(mb_custom, "data")
plots_dir = os.path.join(mb_custom, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Load JSON summary data
summary_file = os.path.join(data_dir, "trajectories_summary.json")
if not os.path.exists(summary_file):
    print(f"Error: {summary_file} not found. Run simulations first.")
    exit(1)

with open(summary_file, "r") as f:
    data = json.load(f)

d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
colors = {2.0: "#d62728", 4.0: "#ff7f0e", 6.0: "#2ca02c", 8.0: "#1f77b4", 10.0: "#9467bd"}

# Helper to compute velocity from trajectory
def get_vz(traj, times):
    z_start = traj[0][2]
    z_end = traj[-1][2]
    return abs((z_end - z_start) / (times[-1] - times[0]))

def get_vx(traj, times):
    x_start = traj[0][0]
    x_end = traj[-1][0]
    return abs((x_end - x_start) / (times[-1] - times[0]))

# =========================================================================
# 1. SPHERE PAIR PLOTS
# =========================================================================
print("Plotting Sphere Pair Trajectories...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Single sphere velocity
v0_sphere = get_vz(data["sphere"]["single"]["0.0"]["traj_b1"], data["sphere"]["single"]["0.0"]["times"])

# Panel A: (x, z) spatial trajectories
ax = axes[0, 0]
for d_r in d_ratios:
    sim = data["sphere"]["pair"][str(d_r)]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)} (B1)')
    ax.plot(tb2[:, 0], tb2[:, 2], '--', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)} (B2)')
ax.set_title("Spatial Trajectories (x vs z Plane)", fontsize=12, fontweight='bold')
ax.set_xlabel("Lateral Position x", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=8, loc='lower center', ncol=2)

# Panel B: Vertical position z(t) vs time
ax = axes[0, 1]
for d_r in d_ratios:
    sim = data["sphere"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    ax.plot(t, tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)}')
ax.plot(data["sphere"]["single"]["0.0"]["times"], np.array(data["sphere"]["single"]["0.0"]["traj_b1"])[:, 2], 
        'k:', linewidth=2.5, label='Single Sphere (d/R → ∞)')
ax.set_title("Vertical Trajectory z(t) vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel C: Separation distance d(t)/R vs time
ax = axes[1, 0]
for d_r in d_ratios:
    sim = data["sphere"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    dist = np.sqrt(np.sum((tb1[:, :3] - tb2[:, :3])**2, axis=1))
    ax.plot(t, dist, '-', color=colors[d_r], linewidth=2.0, label=f'Init d/R = {int(d_r)}')
ax.set_title("Separation Distance d(t)/R vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Instantaneous Separation d(t) / R", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel D: Velocity enhancement ratio V_z / V_0 vs d/R
ax = axes[1, 1]
enhancement = []
for d_r in d_ratios:
    sim = data["sphere"]["pair"][str(d_r)]
    vz = get_vz(sim["traj_b1"], sim["times"])
    enhancement.append(vz / v0_sphere)

# Batchelor / Stimson-Jeffery Stokeslet asymptotic prediction: 1 + 0.75*(R/d)
d_theory = np.linspace(1.8, 11.0, 100)
theory_v = 1.0 + 0.75 * (1.0 / d_theory)

ax.plot(d_ratios, enhancement, 'ro-', linewidth=2.2, markersize=8, label='Forward Euler Simulations')
ax.plot(d_theory, theory_v, 'k--', linewidth=1.8, label=r'Stokeslet Theory ($1 + \frac{3}{4}\frac{R}{d}$)')
ax.axhline(1.0, color='gray', linestyle=':', label='Single Sphere Limit (1.0)')
ax.set_title("Sedimentation Velocity Enhancement (|V_z| / V_0)", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Normalized Velocity |V_z| / V_0", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

plt.tight_layout()
p_sphere = os.path.join(plots_dir, "sphere_pair_trajectories.png")
plt.savefig(p_sphere, dpi=300)
plt.close()
print(f"Saved: {p_sphere}")

# =========================================================================
# 2. ELLIPSOID PAIR PLOTS (Different Angles 0, 45, 90 deg)
# =========================================================================
print("Plotting Ellipsoid Pair Trajectories...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
angles = [0, 45, 90]
ang_colors = {0: "#1f77b4", 45: "#d62728", 90: "#2ca02c"}

# Panel A: Spatial trajectories for d/R = 4 across angles
ax = axes[0, 0]
for ang in angles:
    sim = data["ellipsoid"][f"ang_{ang}"]["4.0"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 2], '-', color=ang_colors[ang], linewidth=2.2, label=f'θ = {ang}° (Body 1)')
    ax.plot(tb2[:, 0], tb2[:, 2], '--', color=ang_colors[ang], linewidth=2.2, label=f'θ = {ang}° (Body 2)')
ax.set_title("Spatial Trajectories (x vs z) at d/R = 4.0", fontsize=12, fontweight='bold')
ax.set_xlabel("Lateral Position x", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel B: z(t) for all angles at d/R = 4
ax = axes[0, 1]
for ang in angles:
    for d_r in [2.0, 6.0, 10.0]:
        sim = data["ellipsoid"][f"ang_{ang}"][str(d_r)]
        tb1 = np.array(sim["traj_b1"])
        ls = '-' if d_r == 2.0 else ('--' if d_r == 6.0 else ':')
        ax.plot(sim["times"], tb1[:, 2], linestyle=ls, color=ang_colors[ang], linewidth=1.8, 
                label=f'θ = {ang}°, d/R={int(d_r)}' if d_r == 2.0 else None)
ax.set_title("Vertical Sedimentation z(t) vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel C: Terminal velocity |V_z| vs d/R for each angle
ax = axes[1, 0]
for ang in angles:
    vels = [get_vz(data["ellipsoid"][f"ang_{ang}"][str(d_r)]["traj_b1"], data["ellipsoid"][f"ang_{ang}"][str(d_r)]["times"]) for d_r in d_ratios]
    v_single = get_vz(data["ellipsoid"][f"ang_{ang}"]["0.0"]["traj_b1"], data["ellipsoid"][f"ang_{ang}"]["0.0"]["times"])
    ax.plot(d_ratios, vels, 'o-', color=ang_colors[ang], linewidth=2.0, markersize=7, label=f'θ = {ang}° (Single: {v_single:.1f})')
    ax.axhline(v_single, color=ang_colors[ang], linestyle=':', alpha=0.7)
ax.set_title("Vertical Velocity |V_z| vs Separation d/R", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Terminal Velocity |V_z|", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel D: Lateral drift velocity |V_x| vs d/R (specifically for 45 deg tilt)
ax = axes[1, 1]
for ang in angles:
    vx_vals = [get_vx(data["ellipsoid"][f"ang_{ang}"][str(d_r)]["traj_b1"], data["ellipsoid"][f"ang_{ang}"][str(d_r)]["times"]) for d_r in d_ratios]
    ax.plot(d_ratios, vx_vals, 's-', color=ang_colors[ang], linewidth=2.0, markersize=7, label=f'θ = {ang}°')
ax.set_title("Lateral Drift Velocity |V_x| vs Separation d/R", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Lateral Drift Velocity |V_x|", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

plt.tight_layout()
p_ellip = os.path.join(plots_dir, "ellipsoid_pair_trajectories.png")
plt.savefig(p_ellip, dpi=300)
plt.close()
print(f"Saved: {p_ellip}")

# =========================================================================
# 3. DISC PAIR PLOTS
# =========================================================================
print("Plotting Disc Pair Trajectories...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
v0_disc = get_vz(data["disc"]["single"]["0.0"]["traj_b1"], data["disc"]["single"]["0.0"]["times"])

# Panel A: (x, z) trajectories
ax = axes[0, 0]
for d_r in d_ratios:
    sim = data["disc"]["pair"][str(d_r)]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)}')
    ax.plot(tb2[:, 0], tb2[:, 2], '--', color=colors[d_r], linewidth=2.0)
ax.set_title("Spatial Trajectories (x vs z) for Disc Pairs", fontsize=12, fontweight='bold')
ax.set_xlabel("Lateral Position x", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel B: z(t) vs time
ax = axes[0, 1]
for d_r in d_ratios:
    sim = data["disc"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    ax.plot(t, tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)}')
ax.plot(data["disc"]["single"]["0.0"]["times"], np.array(data["disc"]["single"]["0.0"]["traj_b1"])[:, 2], 
        'k:', linewidth=2.5, label='Single Disc (Isolated)')
ax.set_title("Vertical Displacement z(t) vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel C: Separation d(t)/R vs time
ax = axes[1, 0]
for d_r in d_ratios:
    sim = data["disc"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    dist = np.sqrt(np.sum((tb1[:, :3] - tb2[:, :3])**2, axis=1))
    ax.plot(t, dist, '-', color=colors[d_r], linewidth=2.0, label=f'Init d/R = {int(d_r)}')
ax.set_title("Pair Separation Distance d(t)/R vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Separation d(t) / R", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel D: Normalized velocity |V_z| / V_0
ax = axes[1, 1]
disc_enh = [get_vz(data["disc"]["pair"][str(d_r)]["traj_b1"], data["disc"]["pair"][str(d_r)]["times"]) / v0_disc for d_r in d_ratios]
ax.plot(d_ratios, disc_enh, 'bo-', linewidth=2.2, markersize=8, label='Disc Pair Simulations')
ax.axhline(1.0, color='gray', linestyle=':', label='Single Disc Limit (1.0)')
ax.set_title("Velocity Enhancement (|V_z| / V_0) vs d/R", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Normalized Velocity |V_z| / V_0", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

plt.tight_layout()
p_disc = os.path.join(plots_dir, "disc_pair_trajectories.png")
plt.savefig(p_disc, dpi=300)
plt.close()
print(f"Saved: {p_disc}")

# =========================================================================
# 4. CYLINDER PAIR PLOTS
# =========================================================================
print("Plotting Cylinder Pair Trajectories...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
v0_cyl = get_vz(data["cylinder"]["single"]["0.0"]["traj_b1"], data["cylinder"]["single"]["0.0"]["times"])

# Panel A: (x, z) trajectories
ax = axes[0, 0]
for d_r in d_ratios:
    sim = data["cylinder"]["pair"][str(d_r)]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)}')
    ax.plot(tb2[:, 0], tb2[:, 2], '--', color=colors[d_r], linewidth=2.0)
ax.set_title("Spatial Trajectories (x vs z) for Cylinder Pairs", fontsize=12, fontweight='bold')
ax.set_xlabel("Lateral Position x", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel B: z(t) vs time
ax = axes[0, 1]
for d_r in d_ratios:
    sim = data["cylinder"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    ax.plot(t, tb1[:, 2], '-', color=colors[d_r], linewidth=2.0, label=f'd/R = {int(d_r)}')
ax.plot(data["cylinder"]["single"]["0.0"]["times"], np.array(data["cylinder"]["single"]["0.0"]["traj_b1"])[:, 2], 
        'k:', linewidth=2.5, label='Single Cylinder (Isolated)')
ax.set_title("Vertical Displacement z(t) vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel C: Separation d(t)/R vs time
ax = axes[1, 0]
for d_r in d_ratios:
    sim = data["cylinder"]["pair"][str(d_r)]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    dist = np.sqrt(np.sum((tb1[:, :3] - tb2[:, :3])**2, axis=1))
    ax.plot(t, dist, '-', color=colors[d_r], linewidth=2.0, label=f'Init d/R = {int(d_r)}')
ax.set_title("Separation Distance d(t)/R vs Time", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Separation d(t) / R", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel D: Normalized velocity |V_z| / V_0
ax = axes[1, 1]
cyl_enh = [get_vz(data["cylinder"]["pair"][str(d_r)]["traj_b1"], data["cylinder"]["pair"][str(d_r)]["times"]) / v0_cyl for d_r in d_ratios]
ax.plot(d_ratios, cyl_enh, 'mo-', linewidth=2.2, markersize=8, label='Cylinder Pair Simulations')
ax.axhline(1.0, color='gray', linestyle=':', label='Single Cylinder Limit (1.0)')
ax.set_title("Velocity Enhancement (|V_z| / V_0) vs d/R", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Normalized Velocity |V_z| / V_0", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

plt.tight_layout()
p_cyl = os.path.join(plots_dir, "cylinder_pair_trajectories.png")
plt.savefig(p_cyl, dpi=300)
plt.close()
print(f"Saved: {p_cyl}")

# =========================================================================
# 5. BOOMERANG PAIR PLOTS (Relative Angles 0, 45, 90, 180 deg)
# =========================================================================
print("Plotting Boomerang Pair Trajectories...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
rel_angles = [0, 45, 90, 180]
rel_colors = {0: "#1f77b4", 45: "#ff7f0e", 90: "#2ca02c", 180: "#d62728"}
v0_boom = get_vz(data["boomerang"]["single"]["0.0"]["traj_b1"], data["boomerang"]["single"]["0.0"]["times"])

# Panel A: (x, z) trajectory paths for d/R = 4 across relative angles
ax = axes[0, 0]
for rel_ang in rel_angles:
    sim = data["boomerang"][f"rel_{rel_ang}"]["4.0"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 2], '-', color=rel_colors[rel_ang], linewidth=2.0, label=f'Δφ = {rel_ang}° (B1)')
    ax.plot(tb2[:, 0], tb2[:, 2], '--', color=rel_colors[rel_ang], linewidth=2.0, label=f'Δφ = {rel_ang}° (B2)')
ax.set_title("Planar Trajectories (x vs z) for Boomerang Pairs (d/R = 4)", fontsize=12, fontweight='bold')
ax.set_xlabel("Lateral Position x", fontsize=11)
ax.set_ylabel("Vertical Position z", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=8, ncol=2)

# Panel B: Lateral (x, y) drift showing chiral rotation coupling
ax = axes[0, 1]
for rel_ang in rel_angles:
    sim = data["boomerang"][f"rel_{rel_ang}"]["4.0"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    ax.plot(tb1[:, 0], tb1[:, 1], '-', color=rel_colors[rel_ang], linewidth=2.0, label=f'Δφ = {rel_ang}° (B1)')
    ax.plot(tb2[:, 0], tb2[:, 1], '--', color=rel_colors[rel_ang], linewidth=2.0, label=f'Δφ = {rel_ang}° (B2)')
ax.set_title("Horizontal Plane (x vs y) Chiral Drift (d/R = 4)", fontsize=12, fontweight='bold')
ax.set_xlabel("Position x", fontsize=11)
ax.set_ylabel("Position y", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=8, ncol=2)

# Panel C: Separation distance d(t)/R vs time across relative angles
ax = axes[1, 0]
for rel_ang in rel_angles:
    sim = data["boomerang"][f"rel_{rel_ang}"]["4.0"]
    t = sim["times"]
    tb1 = np.array(sim["traj_b1"])
    tb2 = np.array(sim["traj_b2"])
    dist = np.sqrt(np.sum((tb1[:, :3] - tb2[:, :3])**2, axis=1))
    ax.plot(t, dist, '-', color=rel_colors[rel_ang], linewidth=2.2, label=f'Δφ = {rel_ang}°')
ax.set_title("Pair Separation Distance d(t)/R vs Time (Init d/R = 4.0)", fontsize=12, fontweight='bold')
ax.set_xlabel("Time t (s)", fontsize=11)
ax.set_ylabel("Separation d(t) / R", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

# Panel D: Sedimentation velocity |V_z| vs d/R across relative angles
ax = axes[1, 1]
for rel_ang in rel_angles:
    vz_vals = [get_vz(data["boomerang"][f"rel_{rel_ang}"][str(d_r)]["traj_b1"], data["boomerang"][f"rel_{rel_ang}"][str(d_r)]["times"]) for d_r in d_ratios]
    ax.plot(d_ratios, vz_vals, 'o-', color=rel_colors[rel_ang], linewidth=2.0, markersize=7, label=f'Δφ = {rel_ang}°')
ax.axhline(v0_boom, color='gray', linestyle=':', label='Single Boomerang Limit')
ax.set_title("Sedimentation Velocity |V_z| vs Separation Ratio", fontsize=12, fontweight='bold')
ax.set_xlabel("Separation Ratio (d / R)", fontsize=11)
ax.set_ylabel("Terminal Velocity |V_z|", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=9)

plt.tight_layout()
p_boom = os.path.join(plots_dir, "boomerang_pair_trajectories.png")
plt.savefig(p_boom, dpi=300)
plt.close()
print(f"Saved: {p_boom}")

# =========================================================================
# 6. MASTER SUMMARY FIGURE: Comparative Scaling Across All 5 Geometries
# =========================================================================
print("Plotting Master Summary Interaction Scaling...")
plt.figure(figsize=(10, 6.5))

geom_labels = {
    "sphere": ("Sphere (Pair)", "ro-", "#d62728"),
    "ellipsoid_0": ("Ellipsoid (θ=0°)", "s-", "#1f77b4"),
    "ellipsoid_45": ("Ellipsoid (θ=45°)", "^-", "#ff7f0e"),
    "ellipsoid_90": ("Ellipsoid (θ=90°)", "v-", "#2ca02c"),
    "disc": ("Disc (Broadside)", "p-", "#9467bd"),
    "cylinder": ("Cylinder (Pair)", "d-", "#8c564b"),
    "boomerang_0": ("Boomerang (Δφ=0°)", "*-", "#e377c2"),
    "boomerang_90": ("Boomerang (Δφ=90°)", "x--", "#17becf")
}

# Collect normalized velocities
curves = {}
curves["sphere"] = [get_vz(data["sphere"]["pair"][str(d_r)]["traj_b1"], data["sphere"]["pair"][str(d_r)]["times"]) / v0_sphere for d_r in d_ratios]
for ang in [0, 45, 90]:
    v0_e = get_vz(data["ellipsoid"][f"ang_{ang}"]["0.0"]["traj_b1"], data["ellipsoid"][f"ang_{ang}"]["0.0"]["times"])
    curves[f"ellipsoid_{ang}"] = [get_vz(data["ellipsoid"][f"ang_{ang}"][str(d_r)]["traj_b1"], data["ellipsoid"][f"ang_{ang}"][str(d_r)]["times"]) / v0_e for d_r in d_ratios]
curves["disc"] = disc_enh
curves["cylinder"] = cyl_enh
curves["boomerang_0"] = [get_vz(data["boomerang"]["rel_0"][str(d_r)]["traj_b1"], data["boomerang"]["rel_0"][str(d_r)]["times"]) / v0_boom for d_r in d_ratios]
curves["boomerang_90"] = [get_vz(data["boomerang"]["rel_90"][str(d_r)]["traj_b1"], data["boomerang"]["rel_90"][str(d_r)]["times"]) / v0_boom for d_r in d_ratios]

for k, (lbl, style, col) in geom_labels.items():
    plt.plot(d_ratios, curves[k], style, color=col, linewidth=2.0, markersize=8, label=lbl)

# Add Stokeslet theoretical decay
d_th = np.linspace(1.8, 10.5, 100)
plt.plot(d_th, 1.0 + 0.75 * (1.0 / d_th), 'k--', linewidth=2.0, label=r'Stokeslet Asymptotic ($1 + \frac{3}{4}\frac{R}{d}$)')
plt.axhline(1.0, color='gray', linestyle=':', label='Isolated Single Body (1.0)')

plt.title("Hydrodynamic Interaction Decay (|V_z| / V_0) Across Particle Geometries", fontsize=13, fontweight='bold')
plt.xlabel("Initial Separation Ratio (d / R)", fontsize=12)
plt.ylabel("Sedimentation Velocity Enhancement (|V_z| / V_0)", fontsize=12)
plt.xticks(d_ratios, labels=[f"d/R={int(d)}" for d in d_ratios])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=9, loc='upper right', ncol=2)

plt.tight_layout()
p_summary = os.path.join(plots_dir, "summary_pair_interaction_scaling.png")
plt.savefig(p_summary, dpi=300)
plt.close()
print(f"Saved: {p_summary}")

print("\n" + "=" * 80)
print("ALL MULTI-BODY TRAJECTORY PLOTS GENERATED SUCCESSFULLY!")
print("=" * 80)
