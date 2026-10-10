import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Cylinder Mobility Scheme velocity vs force plotter")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

N = 86
blob_radius = 0.0742
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
cyl_dir = os.path.join(base_dir, "custom_simulations", "cylinder", "single")
mob_dir = os.path.join(cyl_dir, "mobility")
struct_dir = os.path.join(cyl_dir, "structures")
data_dir = os.path.join(mob_dir, "data")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

v_src = os.path.join(struct_dir, "cylinder_N_86.vertex")
v_dst = os.path.join(mob_dir, "cylinder_N_86.vertex")
if not os.path.exists(v_dst):
    import shutil
    shutil.copy(v_src, v_dst)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

# Test two orientations:
# 1. Broadside (perpendicular to gravity): orientation q = (1, 0, 0, 0)
# 2. Parallel (vertical cylinder, tilted 90 deg around y): q = (cos(pi/4), 0, sin(pi/4), 0)
orientations = {
    "Broadside (Perpendicular)": (1.0, 0.0, 0.0, 0.0),
    "Parallel (End-on)": (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
}

results = {"Broadside (Perpendicular)": [], "Parallel (End-on)": []}

print("=" * 80)
print("CYLINDER BENCHMARK: VELOCITY VS FORCE (MOBILITY SCHEME)")
print(f"Discretization: N={N}, Blob Radius: a={blob_radius:.4f}")
print("=" * 80)

for ori_name, q in orientations.items():
    tag = "broadside" if "Broadside" in ori_name else "parallel"
    clones_file = os.path.join(mob_dir, f"single_{tag}.clones")
    with open(clones_file, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        
    for F in F_values:
        g = F / N
        out_name = f"data/cylinder_{tag}_F_{int(F)}"
        inp_file = os.path.join(mob_dir, f"input_{tag}_F_{int(F)}.dat")
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
structure cylinder_N_86.vertex single_{tag}.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_{tag}_F_{int(F)}.dat"]
            print(f"[Mobility] Running Cylinder {tag}, F={F}...", flush=True)
            subprocess.run(cmd, cwd=mob_dir, env=env, check=True, capture_output=True)
            
        vel_data = np.loadtxt(vel_file)
        vz = abs(vel_data[2])
        results[ori_name].append((F, vz))

print("\nAll mobility calculations complete! Generating diagnostic plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Velocity vs Force
ax1 = axes[0]
colors = {"Broadside (Perpendicular)": "#2ca02c", "Parallel (End-on)": "#1f77b4"}
markers = {"Broadside (Perpendicular)": "s-", "Parallel (End-on)": "o-"}

for ori_name, pts in results.items():
    fs = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    ax1.plot(fs, vs, markers[ori_name], color=colors[ori_name], linewidth=2.2, markersize=8,
             label=f"{ori_name} (Mobility = {vs[0]/fs[0]:.4f})")

ax1.set_title("Cylinder Terminal Velocity vs Force (Mobility Scheme)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Applied Force F", fontsize=11)
ax1.set_ylabel("Terminal Settling Velocity |V_z|", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Panel 2: Orientation Drag Anisotropy Ratio
ax2 = axes[1]
v_broad = [p[1] for p in results["Broadside (Perpendicular)"]]
v_para = [p[1] for p in results["Parallel (End-on)"]]
ratios = [vp / vb for vp, vb in zip(v_para, v_broad)]

ax2.plot(F_values, ratios, "md-", linewidth=2.2, markersize=8, label="Numerical V_para / V_perp")
ax2.axhline(2.0, color="crimson", linestyle="--", linewidth=1.8, label="Slender-Body Limit (V_para / V_perp -> 2.0)")
for f, r in zip(F_values, ratios):
    ax2.text(f, r + 0.02, f"{r:.3f}", ha="center", fontsize=9, fontweight="bold")

ax2.set_ylim(1.2, 2.2)
ax2.set_title("Hydrodynamic Anisotropy Ratio (V_parallel / V_perpendicular)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Applied Force F", fontsize=11)
ax2.set_ylabel("Velocity Ratio", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()

p1 = os.path.join(mob_dir, "cylinder_velocity_vs_force.png")
p2 = os.path.join(thesis_fig_dir, "cylinder_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved Cylinder Velocity vs Force plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")
