import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Boomerang Mobility Scheme benchmark across N=7, 11, 15")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

Ns = [7, 11, 15]
blob_radii = {
    7: 0.5250,
    11: 0.4000,
    15: 0.3245
}
F_values = [300.0, 600.0, 900.0, 1200.0, 1500.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
boom_dir = os.path.join(base_dir, "custom_simulations", "boomerang", "single")
mob_dir = os.path.join(boom_dir, "mobility")
struct_dir = os.path.join(boom_dir, "structures")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

colors = {7: "#ff7f0e", 11: "#2ca02c", 15: "#d62728"}
markers = {7: "o", 11: "s", 15: "^"}

results = {}

print("=" * 80)
print("BOOMERANG RESOLUTION BENCHMARK: MOBILITY SCHEME ACROSS N in [7, 11, 15]")
print("=" * 80)

for N in Ns:
    n_dir = os.path.join(mob_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    v_src = os.path.join(struct_dir, f"boomerang_N_{N}.vertex")
    v_dst = os.path.join(n_dir, f"boomerang_N_{N}.vertex")
    if not os.path.exists(v_dst):
        import shutil
        shutil.copyfile(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    brad = blob_radii[N]
    forces, vels, omegas = [], [], []
    
    for F in F_values:
        g = F / N
        out_name = f"data/boomerang_N_{N}_F_{int(F)}"
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
structure boomerang_N_{N}.vertex single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Mobility] Running Boomerang N={N}, F={F}...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
            
        vel_data = np.loadtxt(vel_file)
        vz = abs(vel_data[2])
        omega_mag = np.linalg.norm(vel_data[3:6])
        forces.append(F)
        vels.append(vz)
        omegas.append(omega_mag)
        
    results[N] = {
        "forces": forces,
        "velocities": vels,
        "omegas": omegas,
        "mu_T": vels[0] / forces[0],
        "mu_TR": omegas[0] / forces[0],
        "brad": brad
    }

print("\nGenerating diagnostic 4-panel figure for Boomerang...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel A: Translational Velocity vs Force
ax_a = axes[0, 0]
for N in Ns:
    d = results[N]
    ax_a.plot(d["forces"], d["velocities"], f"-{markers[N]}", color=colors[N], linewidth=2.0, markersize=8,
              label=f"N = {N} (μ_T = {d['mu_T']:.4f}, a = {d['brad']:.3f})")
ax_a.set_title("Translational Settling Velocity |V_z| vs Force", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Applied Force F", fontsize=11)
ax_a.set_ylabel("Settling Velocity |V_z|", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=10)

# Panel B: Induced Angular Velocity |Omega| vs Force
ax_b = axes[0, 1]
for N in Ns:
    d = results[N]
    ax_b.plot(d["forces"], d["omegas"], f"-{markers[N]}", color=colors[N], linewidth=2.0, markersize=8,
              label=f"N = {N} (μ_TR = {d['mu_TR']:.4e})")
ax_b.set_title("Chiral Rotation Coupling (|Ω| vs Force)", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Applied Force F", fontsize=11)
ax_b.set_ylabel("Induced Angular Velocity |Ω| (rad/s)", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=10)

# Panel C: Chiral Coupling Ratio (|Omega| / |V_z|) vs N
ax_c = axes[1, 0]
coupling_ratios = [results[N]["mu_TR"] / results[N]["mu_T"] for N in Ns]
ax_c.plot(Ns, coupling_ratios, "md-", linewidth=2.2, markersize=8, label="Chiral Coupling Ratio μ_TR / μ_T")
for N, cr in zip(Ns, coupling_ratios):
    ax_c.text(N, cr + 0.0005, f"{cr:.4f}", ha="center", fontsize=9, fontweight="bold")
ax_c.set_title("Chiral Coupling Ratio (μ_TR / μ_T) vs N", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Resolution N", fontsize=11)
ax_c.set_ylabel("Ratio μ_TR / μ_T", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=10)

# Panel D: 2D Projection of the 3 Boomerang Meshes
ax_d = axes[1, 1]
sub_offsets = {7: 1.5, 11: 0.0, 15: -1.5}
for N in Ns:
    y_off = sub_offsets[N]
    v_file = os.path.join(struct_dir, f"boomerang_N_{N}.vertex")
    with open(v_file, "r") as f:
        clean_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    pts = np.array([[float(p) for p in l.split()] for l in clean_lines[1:]])
    ax_d.plot(pts[:, 0], pts[:, 1] + y_off, "o-", color=colors[N], markersize=5.0, label=f"N={N}")
    ax_d.text(2.3, y_off + 0.5, f"N = {N} (a={results[N]['brad']:.3f})\nμ_T={results[N]['mu_T']:.4f}\nμ_TR={results[N]['mu_TR']:.4e}",
              va="center", ha="left", fontsize=9, fontweight="bold")

ax_d.set_xlim(-0.5, 4.0)
ax_d.set_ylim(-2.0, 3.5)
ax_d.set_title("Boomerang Discretization Meshes (N=7, 11, 15)", fontsize=12, fontweight="bold")
ax_d.set_xlabel("Arm 1 (x Coordinate)", fontsize=11)
ax_d.set_ylabel("Arm 2 (y Coordinate + Offset)", fontsize=11)
ax_d.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

p1 = os.path.join(mob_dir, "boomerang_velocity_vs_force.png")
p2 = os.path.join(thesis_fig_dir, "boomerang_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved Boomerang Velocity vs Force plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")

# Print summary table
print("\n" + "=" * 80)
print(f"{'N':<8} {'Blob Radius a':<15} {'V(F=300)':<12} {'mu_T (V/F)':<15} {'mu_TR (Omega/F)':<15}")
print("-" * 80)
for N in Ns:
    d = results[N]
    print(f"{N:<8} {d['brad']:<15.4f} {d['velocities'][0]:<12.2f} {d['mu_T']:<15.5f} {d['mu_TR']:<15.4e}")
print("=" * 80)
