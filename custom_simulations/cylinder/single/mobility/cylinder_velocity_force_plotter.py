import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Cylinder Mobility Scheme velocity vs force plotter across N")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

Ns = [14, 44, 86, 324]
blob_radii = {
    14: 0.1832,
    44: 0.1250,
    86: 0.0742,
    324: 0.0388
}
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
cyl_dir = os.path.join(base_dir, "custom_simulations", "cylinder", "single")
mob_dir = os.path.join(cyl_dir, "mobility")
struct_dir = os.path.join(cyl_dir, "structures")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

colors = {14: "#1f77b4", 44: "#ff7f0e", 86: "#2ca02c", 324: "#d62728"}
markers = {14: "o", 44: "s", 86: "^", 324: "D"}

results = {}

print("=" * 80)
print("CYLINDER RESOLUTION BENCHMARK: VELOCITY VS FORCE (MOBILITY SCHEME)")
print(f"Testing N in {Ns}")
print("=" * 80)

for N in Ns:
    n_dir = os.path.join(mob_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    v_src = os.path.join(struct_dir, f"cylinder_N_{N}.vertex")
    v_dst = os.path.join(n_dir, f"cylinder_N_{N}.vertex")
    if not os.path.exists(v_dst):
        import shutil
        shutil.copyfile(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    brad = blob_radii[N]
    forces, vels = [], []
    
    for F in F_values:
        g = F / N
        out_name = f"data/cylinder_N_{N}_F_{int(F)}"
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        vel_file = os.path.join(n_dir, f"{out_name}.velocity.dat")
        
        inp_content = f"""scheme                                   mobility
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
eta                                      1.0
g                                        {g}
blob_radius                              {brad:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
output_name                              {out_name}
structure cylinder_N_{N}.vertex single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Mobility] Running Cylinder N={N}, F={F}...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
            
        vel_data = np.loadtxt(vel_file)
        vz = abs(vel_data[2])
        forces.append(F)
        vels.append(vz)
        
    results[N] = {
        "forces": forces,
        "velocities": vels,
        "mobility": vels[0] / forces[0],
        "brad": brad
    }

print("\nGenerating diagnostic 4-panel figure for Cylinder...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel A: Terminal Velocity vs Force
ax_a = axes[0, 0]
for N in Ns:
    d = results[N]
    ax_a.plot(d["forces"], d["velocities"], f"-{markers[N]}", color=colors[N], linewidth=2.0, markersize=8,
              label=f"N = {N} (μ = {d['mobility']:.4f}, a = {d['brad']:.4f})")
ax_a.set_title("Cylinder Terminal Velocity vs Force Across N", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Applied Force F", fontsize=11)
ax_a.set_ylabel("Terminal Velocity |V_z|", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=10)

# Panel B: Mobility μ = V/F vs Discretization Resolution N
ax_b = axes[0, 1]
mob_vals = [results[N]["mobility"] for N in Ns]
ax_b.plot(Ns, mob_vals, "bo-", linewidth=2.2, markersize=8, label="Numerical Mobility μ = V/F")
for N, m in zip(Ns, mob_vals):
    ax_b.annotate(f"N={N}\nμ={m:.4f}", xy=(N, m), xytext=(N, m + 0.003),
                  ha="center", fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))
ax_b.set_title("Mobility μ Convergence vs Number of Blobs N", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Resolution N", fontsize=11)
ax_b.set_ylabel("Hydrodynamic Mobility μ (V / F)", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=10)

# Panel C: Relative Drag Convergence (%) relative to finest mesh N=324
ax_c = axes[1, 0]
ref_mob = results[324]["mobility"]
rel_diff = [abs(results[N]["mobility"] - ref_mob) / ref_mob * 100.0 for N in Ns]
ax_c.plot(Ns, rel_diff, "rs-", linewidth=2.2, markersize=8, label="% Difference vs N=324")
for N, diff in zip(Ns, rel_diff):
    ax_c.text(N, diff + 1.0, f"{diff:.2f}%", ha="center", fontsize=9, fontweight="bold")
ax_c.set_title("Convergence Difference Relative to Fine Mesh (N=324)", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Resolution N", fontsize=11)
ax_c.set_ylabel("Difference (%)", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=10)

# Panel D: Visual 3D/2D Projection of the 4 Meshes
ax_d = axes[1, 1]
sub_offsets = {14: 1.5, 44: 0.5, 86: -0.5, 324: -1.5}
for N in Ns:
    y_off = sub_offsets[N]
    v_file = os.path.join(struct_dir, f"cylinder_N_{N}.vertex")
    with open(v_file, "r") as f:
        pts = np.array([[float(p) for p in l.split()] for l in f.readlines()[1:] if l.strip()])
    ax_d.plot(pts[:, 0], pts[:, 1] + y_off, ".", color=colors[N], markersize=3.0, label=f"N={N}")
    ax_d.text(1.2, y_off, f"N = {N} (a={results[N]['brad']:.4f})\nμ = {results[N]['mobility']:.4f}",
              va="center", ha="left", fontsize=9, fontweight="bold")

ax_d.set_xlim(-1.3, 2.3)
ax_d.set_ylim(-2.2, 2.2)
ax_d.set_title("Cylinder Discretization Meshes (Side Projection)", fontsize=12, fontweight="bold")
ax_d.set_xlabel("Axial Coordinate x", fontsize=11)
ax_d.set_ylabel("Radial Coordinate y (Offset)", fontsize=11)
ax_d.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

p1 = os.path.join(mob_dir, "cylinder_velocity_vs_force.png")
p2 = os.path.join(thesis_fig_dir, "cylinder_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved Cylinder Velocity vs Force plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")

# Print summary table
print("\n" + "=" * 80)
print(f"{'N':<8} {'Blob Radius a':<15} {'V(F=1000)':<12} {'Mobility mu':<15} {'Diff vs N=324 (%)':<20}")
print("-" * 80)
for N in Ns:
    d = results[N]
    diff = abs(d["mobility"] - ref_mob) / ref_mob * 100.0
    print(f"{N:<8} {d['brad']:<15.4f} {d['velocities'][0]:<12.2f} {d['mobility']:<15.5f} {diff:<20.2f}%")
print("=" * 80)
