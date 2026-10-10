import os
import sys
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
disc_dir = os.path.join(base_dir, "custom_simulations", "disc")
angles_dir = os.path.join(disc_dir, "angles")
data_dir = os.path.join(angles_dir, "data")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

# Discretization parameters: N=37 disc
N = 37
blob_radius = 1.0 / (2.0 * 3.0) # 0.1667 for m=3 rings
F = 2000.0
g = F / N
dt = 0.05
n_steps = 100
T = dt * n_steps # 5.0 s

# Copy disc_N_37.vertex into angles directory
v_src = os.path.join(disc_dir, "structures", "disc_N_37.vertex")
v_dst = os.path.join(angles_dir, "disc_N_37.vertex")
if not os.path.exists(v_dst):
    import shutil
    shutil.copy(v_src, v_dst)

angles_deg = [0, 15, 30, 45, 60, 75, 90]
colors = {
    0: "#1f77b4",
    15: "#ff7f0e",
    30: "#2ca02c",
    45: "#d62728",
    60: "#9467bd",
    75: "#8c564b",
    90: "#e377c2"
}

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

traj_data = {}

print("=" * 80)
print(f"DISC ANGLE INCLINATION & LATERAL DRIFT SIMULATIONS (FORWARD EULER)")
print(f"Discretization: N={N}, Blob Radius: a={blob_radius:.4f}, Force: F={F}, dt={dt}, Steps={n_steps}")
print("=" * 80)

for ang in angles_deg:
    ang_rad = math.radians(ang)
    # Quaternion for pitch rotation around Y-axis by angle ang
    # This tilts the disc normal (Z-axis) towards the X-axis
    qw = math.cos(ang_rad / 2.0)
    qx = 0.0
    qy = math.sin(ang_rad / 2.0)
    qz = 0.0
    
    clones_file = os.path.join(angles_dir, f"clones_angle_{ang}.clones")
    with open(clones_file, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n")
        
    out_name = f"data/disc_angle_{ang}"
    inp_file = os.path.join(angles_dir, f"input_angle_{ang}.dat")
    cfg_file = os.path.join(angles_dir, f"{out_name}.clones_angle_{ang}.config")
    
    inp_content = f"""scheme                                   deterministic_forward_euler
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        {g:.6f}
blob_radius                              {blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_name}
structure disc_N_{N}.vertex clones_angle_{ang}.clones
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_angle_{ang}.dat"]
    print(f"Running Disc Tilt Angle theta = {ang:2d} deg with Forward Euler...", flush=True)
    subprocess.run(cmd, cwd=angles_dir, env=env, check=True, capture_output=True)
    
    # Parse trajectory from config file
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    t_list, x_list, y_list, z_list = [], [], [], []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        pos = bodies[0] # [x, y, z, qw, qx, qy, qz]
        t_list.append(step * dt)
        x_list.append(pos[0])
        y_list.append(pos[1])
        z_list.append(pos[2])
        step += 1
        
    vx = (x_list[-1] - x_list[0]) / T
    vz = (z_list[-1] - z_list[0]) / T
    traj_data[ang] = {
        "times": np.array(t_list),
        "x": np.array(x_list),
        "z": np.array(z_list),
        "vx": vx,
        "vz": vz
    }
    print(f"  -> Finished theta = {ang:2d} deg | Vx = {vx:7.2f} | Vz = {vz:7.2f} | Glide Ratio (|Vx/Vz|) = {abs(vx/vz):.3f}")

print("\nAll Forward Euler trajectory runs complete! Plotting results...")

# =========================================================================
# Diagnostic 4-Panel Visualization
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# -------------------------------------------------------------------------
# Panel A: Spatial Flight Paths (x vs z Plane)
# -------------------------------------------------------------------------
ax_a = axes[0, 0]
for ang in angles_deg:
    d = traj_data[ang]
    style = "-" if ang in [0, 45, 90] else "--"
    lw = 2.5 if ang == 45 else 1.8
    ax_a.plot(d["x"], d["z"], style, color=colors[ang], linewidth=lw,
              label=f"θ = {ang:2d}° (Vx={d['vx']:.1f}, Vz={d['vz']:.1f})")

ax_a.set_title("Spatial Trajectory Flight Paths (x vs z Plane)", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Lateral Displacement x", fontsize=11)
ax_a.set_ylabel("Vertical Displacement z", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=9, loc="lower left")

# Annotation for maximum glide
max_x_45 = traj_data[45]["x"][-1]
max_z_45 = traj_data[45]["z"][-1]
ax_a.annotate("Maximum Lateral Drift\n(θ = 45° Pitch Tilt)", xy=(max_x_45, max_z_45),
              xytext=(max_x_45 + 15, max_z_45 + 150),
              arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
              fontsize=10, fontweight="bold", color="#d62728")

# -------------------------------------------------------------------------
# Panel B: Lateral Displacement x(t) vs Time
# -------------------------------------------------------------------------
ax_b = axes[0, 1]
for ang in angles_deg:
    d = traj_data[ang]
    style = "-" if ang in [0, 45, 90] else "--"
    lw = 2.5 if ang == 45 else 1.8
    ax_b.plot(d["times"], d["x"], style, color=colors[ang], linewidth=lw, label=f"θ = {ang:2d}°")

ax_b.set_title("Lateral Shift x(t) vs Time (Forward Euler)", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Time t (s)", fontsize=11)
ax_b.set_ylabel("Lateral Position x", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=9, loc="lower left")

# -------------------------------------------------------------------------
# Panel C: Vertical Displacement z(t) vs Time
# -------------------------------------------------------------------------
ax_c = axes[1, 0]
for ang in angles_deg:
    d = traj_data[ang]
    style = "-" if ang in [0, 45, 90] else "--"
    lw = 2.5 if ang == 45 else 1.8
    ax_c.plot(d["times"], d["z"], style, color=colors[ang], linewidth=lw, label=f"θ = {ang:2d}°")

ax_c.set_title("Vertical Descent z(t) vs Time", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Time t (s)", fontsize=11)
ax_c.set_ylabel("Vertical Position z", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=9, loc="lower left")

# -------------------------------------------------------------------------
# Panel D: Lateral & Vertical Velocities vs Inclination Angle θ
# -------------------------------------------------------------------------
ax_d = axes[1, 1]
ang_array = np.array(angles_deg)
vx_array = np.array([abs(traj_data[a]["vx"]) for a in angles_deg])
vz_array = np.array([abs(traj_data[a]["vz"]) for a in angles_deg])

# Theoretical fits:
# Vx(theta) ~ A * sin(2*theta)
# Vz(theta) ~ V_broad + (V_edge - V_broad) * sin^2(theta)
theta_fine = np.linspace(0, 90, 200)
theta_rad_fine = np.radians(theta_fine)
A_fit = max(vx_array)
vx_theory = A_fit * np.sin(2.0 * theta_rad_fine)

v_broad = vz_array[0]
v_edge = vz_array[-1]
vz_theory = v_broad + (v_edge - v_broad) * (np.sin(theta_rad_fine)**2)

ax_d.plot(theta_fine, vz_theory, "k:", linewidth=2.0, label="Theory Vz ~ V_0 + ΔV sin²(θ)")
ax_d.plot(theta_fine, vx_theory, "r:", linewidth=2.0, label="Theory |Vx| ~ A sin(2θ)")

ax_d.plot(ang_array, vz_array, "bs-", linewidth=2.2, markersize=8, label="Numerical Settling |Vz|")
ax_d.plot(ang_array, vx_array, "ro-", linewidth=2.2, markersize=8, label="Numerical Drift |Vx|")

ax_d.axvline(45.0, color="gray", linestyle="--", alpha=0.7, label="Peak Drift at θ = 45°")
ax_d.set_title("Terminal Velocities vs Inclination Angle θ", fontsize=12, fontweight="bold")
ax_d.set_xlabel("Disc Tilt Angle θ (degrees)", fontsize=11)
ax_d.set_ylabel("Velocity Magnitude", fontsize=11)
ax_d.grid(True, linestyle="--", alpha=0.6)
ax_d.legend(fontsize=9, loc="center left")

plt.tight_layout()

# Save figures
p1 = os.path.join(angles_dir, "disc_lateral_drift_trajectories.png")
p2 = os.path.join(disc_dir, "disc_lateral_drift_trajectories.png")
p3 = os.path.join(thesis_fig_dir, "disc_lateral_drift_trajectories.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.savefig(p3, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved Disc Angle & Drift Trajectory plots:")
print(f"  -> {p1}")
print(f"  -> {p3}")
