import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Boomerang Mobility Scheme velocity vs force plotter")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

N = 15
blob_radius = 0.3245
F_values = [300.0, 600.0, 900.0, 1200.0, 1500.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
boom_dir = os.path.join(base_dir, "custom_simulations", "boomerang", "single")
mob_dir = os.path.join(boom_dir, "mobility")
struct_dir = os.path.join(boom_dir, "structures")
data_dir = os.path.join(mob_dir, "data")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

v_src = os.path.join(struct_dir, "boomerang_N_15.vertex")
v_dst = os.path.join(mob_dir, "boomerang_N_15.vertex")
if not os.path.exists(v_dst):
    import shutil
    shutil.copy(v_src, v_dst)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

clones_file = os.path.join(mob_dir, "single_body.clones")
with open(clones_file, "w") as f:
    f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")

forces = []
velocities = []
angular_velocities = []

print("=" * 80)
print("BOOMERANG BENCHMARK: VELOCITY VS FORCE (MOBILITY SCHEME)")
print(f"Discretization: N={N}, Blob Radius: a={blob_radius:.4f}")
print("=" * 80)

for F in F_values:
    g = F / N
    out_name = f"data/boomerang_F_{int(F)}"
    inp_file = os.path.join(mob_dir, f"input_F_{int(F)}.dat")
    vel_file = os.path.join(mob_dir, f"{out_name}.velocity.dat")
    
    inp_content = f"""scheme                                   mobility
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
eta                                      1.0
g                                        {g}
blob_radius                              {blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
output_name                              {out_name}
structure boomerang_N_15.vertex single_body.clones
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    if args.rerun or not os.path.exists(vel_file):
        cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
        print(f"[Mobility] Running Boomerang F={F}...", flush=True)
        subprocess.run(cmd, cwd=mob_dir, env=env, check=True, capture_output=True)
        
    vel_data = np.loadtxt(vel_file)
    vz = abs(vel_data[2])
    # omega is vel_data[3:6]
    omega_mag = np.linalg.norm(vel_data[3:6])
    forces.append(F)
    velocities.append(vz)
    angular_velocities.append(omega_mag)

print("\nGenerating Boomerang diagnostic figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Translational Velocity vs Force
ax1 = axes[0]
ax1.plot(forces, velocities, "mo-", linewidth=2.2, markersize=8, label=f"Translational |V_z| (μ_T = {velocities[0]/forces[0]:.4f})")
ax1.set_title("Boomerang Settling Velocity vs Force", fontsize=12, fontweight="bold")
ax1.set_xlabel("Applied Force F", fontsize=11)
ax1.set_ylabel("Translational Settling Velocity |V_z|", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Panel 2: Induced Chiral Rotation Rate vs Force
ax2 = axes[1]
ax2.plot(forces, angular_velocities, "cs-", linewidth=2.2, markersize=8,
         label=f"Induced Angular Velocity |Ω| (μ_TR = {angular_velocities[0]/forces[0]:.4e})")
ax2.set_title("Chiral Rotation-Translation Coupling (|Ω| vs F)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Applied Force F", fontsize=11)
ax2.set_ylabel("Induced Angular Velocity |Ω| (rad/s)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()

p1 = os.path.join(mob_dir, "boomerang_velocity_vs_force.png")
p2 = os.path.join(thesis_fig_dir, "boomerang_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved Boomerang Mobility plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")
