import os
import sys
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
boom_dir = os.path.join(base_dir, "custom_simulations", "boomerang", "single")
traj_dir = os.path.join(boom_dir, "trajectories")
data_dir = os.path.join(traj_dir, "data")
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
v_dst = os.path.join(traj_dir, "boomerang_N_15.vertex")
if not os.path.exists(v_dst):
    import shutil
    shutil.copy(v_src, v_dst)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

clones_file = os.path.join(traj_dir, "single_body.clones")
with open(clones_file, "w") as f:
    f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")

out_name = "data/boomerang_single_traj"
inp_file = os.path.join(traj_dir, "input_boomerang_traj.dat")
cfg_file = os.path.join(traj_dir, f"{out_name}.single_body.config")

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
structure boomerang_N_15.vertex single_body.clones
"""
with open(inp_file, "w") as f:
    f.write(inp_content)

print(f"Running Boomerang Trajectory with Forward Euler (N={N}, F={F}, dt={dt}, steps={n_steps})...")
cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", "input_boomerang_traj.dat"]
subprocess.run(cmd, cwd=traj_dir, env=env, check=True, capture_output=True)

with open(cfg_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

idx = 0
times, xs, ys, zs, qws, qxs, qys, qzs = [], [], [], [], [], [], [], []
step = 0
while idx < len(lines):
    nb = int(lines[idx].split()[0])
    idx += 1
    bodies = []
    for _ in range(nb):
        bodies.append([float(x) for x in lines[idx].split()])
        idx += 1
    pos = bodies[0] # [x, y, z, qw, qx, qy, qz]
    times.append(step * dt)
    xs.append(pos[0])
    ys.append(pos[1])
    zs.append(pos[2])
    qws.append(pos[3])
    qxs.append(pos[4])
    qys.append(pos[5])
    qzs.append(pos[6])
    step += 1

print("\nGenerating Boomerang Chiral Trajectory figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: 2D Spatial Path (x vs z)
ax_a = axes[0, 0]
ax_a.plot(xs, zs, "m-", linewidth=2.5, label="Flight Path")
ax_a.plot(xs[0], zs[0], "go", markersize=8, label="Start (0,0)")
ax_a.plot(xs[-1], zs[-1], "rs", markersize=8, label=f"End ({xs[-1]:.2f}, {zs[-1]:.1f})")
ax_a.set_title("Spatial Trajectory (x vs z Plane)", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Lateral Position x", fontsize=11)
ax_a.set_ylabel("Vertical Position z", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=10)

# Panel B: Lateral Drifts x(t) and y(t)
ax_b = axes[0, 1]
ax_b.plot(times, xs, "b-", linewidth=2.0, label="Lateral x(t)")
ax_b.plot(times, ys, "g--", linewidth=2.0, label="Cross-Stream y(t)")
ax_b.set_title("Cross-Stream Lateral Drift vs Time", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Time t (s)", fontsize=11)
ax_b.set_ylabel("Lateral Displacement", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=10)

# Panel C: Vertical Displacement z(t)
ax_c = axes[1, 0]
ax_c.plot(times, zs, "r-", linewidth=2.2, label=f"Vertical z(t) (Vz = {abs(zs[-1]-zs[0])/T:.2f})")
ax_c.set_title("Vertical Descent z(t) vs Time", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Time t (s)", fontsize=11)
ax_c.set_ylabel("Vertical Position z", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=10)

# Panel D: Orientation Quaternion Evolution (Tumbling / Rotation)
ax_d = axes[1, 1]
ax_d.plot(times, qws, "-", color="navy", label="qw (Scalar)")
ax_d.plot(times, qxs, "--", color="darkorange", label="qx")
ax_d.plot(times, qys, "-.", color="green", label="qy")
ax_d.plot(times, qzs, ":", color="purple", label="qz")
ax_d.set_title("Orientation Quaternions vs Time (Chiral Rotation)", fontsize=12, fontweight="bold")
ax_d.set_xlabel("Time t (s)", fontsize=11)
ax_d.set_ylabel("Quaternion Components", fontsize=11)
ax_d.grid(True, linestyle="--", alpha=0.6)
ax_d.legend(fontsize=9, loc="center right")

plt.tight_layout()

p1 = os.path.join(traj_dir, "boomerang_chiral_trajectory.png")
p2 = os.path.join(thesis_fig_dir, "boomerang_chiral_trajectory.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved Boomerang Trajectory plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")
