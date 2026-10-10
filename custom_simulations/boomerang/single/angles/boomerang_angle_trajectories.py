import os
import sys
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
boom_dir = os.path.join(base_dir, "custom_simulations", "boomerang", "single")
angles_dir = os.path.join(boom_dir, "angles")
data_dir = os.path.join(angles_dir, "data")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

N = 15
blob_radius = 0.3245
F = 500.0
g = F / N
dt = 0.05
n_steps = 100
T = dt * n_steps

v_src = os.path.join(boom_dir, "structures", "boomerang_N_15.vertex")
v_dst = os.path.join(angles_dir, "boomerang_N_15.vertex")
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
print("BOOMERANG ANGLE INCLINATION & LATERAL DRIFT SIMULATIONS (FORWARD EULER)")
print(f"Discretization: N={N}, Blob Radius: a={blob_radius:.4f}, Force: F={F}, dt={dt}, Steps={n_steps}")
print("=" * 80)

for ang in angles_deg:
    ang_rad = math.radians(ang)
    qw = math.cos(ang_rad / 2.0)
    qx = 0.0
    qy = math.sin(ang_rad / 2.0)
    qz = 0.0
    
    clones_file = os.path.join(angles_dir, f"clones_angle_{ang}.clones")
    with open(clones_file, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n")
        
    out_name = f"data/boomerang_angle_{ang}"
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
structure boomerang_N_15.vertex clones_angle_{ang}.clones
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    print(f"Running Boomerang angle = {ang} deg...")
    cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_angle_{ang}.dat"]
    subprocess.run(cmd, cwd=angles_dir, env=env, check=True, capture_output=True)
    
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    times, xs, ys, zs = [], [], [], []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        pos = bodies[0]
        times.append(step * dt)
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])
        step += 1
        
    traj_data[ang] = {
        "time": np.array(times),
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "Vx": (xs[-1] - xs[0]) / T,
        "Vz": (zs[-1] - zs[0]) / T,
    }

print("\nAll Boomerang angle simulations completed successfully!")
print("Generating diagnostic 4-panel figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Trajectory (x vs z)
ax_a = axes[0, 0]
for ang in angles_deg:
    d = traj_data[ang]
    ax_a.plot(d["x"], d["z"], "-", color=colors[ang], linewidth=2.0, label=f"{ang} deg")
ax_a.set_title("Trajectory Paths (x vs z) for Varying Angles", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Lateral Position x", fontsize=11)
ax_a.set_ylabel("Vertical Position z", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=9, loc="lower left")

# Panel B: Lateral Drift x(t)
ax_b = axes[0, 1]
for ang in angles_deg:
    d = traj_data[ang]
    ax_b.plot(d["time"], d["x"], "-", color=colors[ang], linewidth=2.0, label=f"{ang} deg (Vx={d['Vx']:.2f})")
ax_b.set_title("Lateral Position x(t) vs Time", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Time t (s)", fontsize=11)
ax_b.set_ylabel("Lateral Position x", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=9, loc="upper left")

# Panel C: Vertical Descent z(t)
ax_c = axes[1, 0]
for ang in angles_deg:
    d = traj_data[ang]
    ax_c.plot(d["time"], d["z"], "-", color=colors[ang], linewidth=2.0, label=f"{ang} deg (Vz={abs(d['Vz']):.2f})")
ax_c.set_title("Vertical Descent z(t) vs Time", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Time t (s)", fontsize=11)
ax_c.set_ylabel("Vertical Position z", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=9, loc="lower left")

# Panel D: Maximum Lateral Drift vs Angle
ax_d = axes[1, 1]
final_drifts = [traj_data[ang]["x"][-1] for ang in angles_deg]
lat_vels = [traj_data[ang]["Vx"] for ang in angles_deg]
ax_d.plot(angles_deg, final_drifts, "ro-", linewidth=2.0, markersize=7, label="Final Lateral Drift x(T)")
ax_d.set_title("Lateral Drift vs Initial Orientation Angle", fontsize=12, fontweight="bold")
ax_d.set_xlabel("Initial Angle (deg)", fontsize=11)
ax_d.set_ylabel("Lateral Displacement x(T)", fontsize=11)
ax_d.grid(True, linestyle="--", alpha=0.6)
ax_d.set_xticks(angles_deg)
ax_d.legend(fontsize=10)

plt.tight_layout()

p1 = os.path.join(angles_dir, "boomerang_lateral_drift_trajectories.png")
p2 = os.path.join(thesis_fig_dir, "boomerang_lateral_drift_trajectories.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved Boomerang Angle Trajectory plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")

print("\n" + "=" * 80)
print(f"{'Angle (deg)':<12} {'Lateral Vx':<15} {'Vertical Vz':<15} {'Glide Ratio |Vx/Vz|':<20} {'x_final':<12}")
print("-" * 80)
for ang in angles_deg:
    d = traj_data[ang]
    glide = abs(d["Vx"] / d["Vz"]) if abs(d["Vz"]) > 1e-6 else 0.0
    print(f"{ang:<12} {d['Vx']:<15.2f} {d['Vz']:<15.2f} {glide:<20.4f} {d['x'][-1]:<12.2f}")
print("=" * 80)
