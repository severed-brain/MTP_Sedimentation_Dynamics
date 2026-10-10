import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser(description="Disc Mobility Scheme velocity vs force plotter")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

# Configuration
Ns = [19, 37, 61, 91]
rings = {19: 2, 37: 3, 61: 4, 91: 5}
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
disc_dir = os.path.join(base_dir, "custom_simulations", "disc")
struct_dir = os.path.join(disc_dir, "structures")
plots_dir = os.path.join(disc_dir, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

results = {}
colors = {19: "#1f77b4", 37: "#ff7f0e", 61: "#2ca02c", 91: "#d62728"}
markers = {19: "o", 37: "s", 61: "^", 91: "D"}

print("=" * 80)
print("DISC BENCHMARK: VELOCITY VS FORCE (MOBILITY SCHEME)")
print("=" * 80)

for N in Ns:
    m = rings[N]
    optimal_blob_radius = 1.0 / (2.0 * m)
    n_dir = os.path.join(disc_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    # Ensure vertex file is present
    v_src = os.path.join(struct_dir, f"disc_N_{N}.vertex")
    v_dst = os.path.join(n_dir, f"disc_N_{N}.vertex")
    if not os.path.exists(v_dst):
        import shutil
        shutil.copy(v_src, v_dst)
        
    # Single body clones file
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0 0 0 1 0 0 0\n")
        
    forces = []
    velocities = []
    
    for F in F_values:
        g = F / N
        out_name = f"data/disc_N_{N}_F_{int(F)}"
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        vel_file = os.path.join(n_dir, f"{out_name}.velocity.dat")
        
        inp_content = f"""scheme                                   mobility
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
eta                                      1.0
g                                        {g}
blob_radius                              {optimal_blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
output_name                              {out_name}
structure disc_N_{N}.vertex single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Mobility] Running Disc N={N} (m={m} rings), F={F} (g={g:.3f})...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
        else:
            print(f"[Mobility] Using cached result for N={N}, F={F}")
            
        vel_data = np.loadtxt(vel_file)
        vz = abs(vel_data[2])
        forces.append(F)
        velocities.append(vz)
        
    results[N] = {
        "m": m,
        "a": optimal_blob_radius,
        "forces": forces,
        "velocities": velocities
    }

print("\nAll mobility calculations complete! Generating diagnostic plots...")

# =========================================================================
# Plotting Diagnostic 4-Panel Figure
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Analytical Stokes reference: V_exact = F / (16 * eta * R) = F / 16 (for eta=1, R=1)
F_fine = np.linspace(500, 5500, 100)
V_exact_fine = F_fine / 16.0

# -------------------------------------------------------------------------
# Panel A: Velocity vs Force
# -------------------------------------------------------------------------
ax_a = axes[0, 0]
ax_a.plot(F_fine, V_exact_fine, "k--", linewidth=2.5, label="Exact Analytical: F / (16 η R)")
for N in Ns:
    data = results[N]
    ax_a.plot(data["forces"], data["velocities"], f"-{markers[N]}", color=colors[N],
              linewidth=2.0, markersize=8, label=f"N = {N} (m = {data['m']}, a = {data['a']:.3f})")

ax_a.set_title("Disc Terminal Velocity vs Applied Force (Broadside Settling)", fontsize=12, fontweight="bold")
ax_a.set_xlabel("Applied Force F (g · N)", fontsize=11)
ax_a.set_ylabel("Terminal Velocity |V_z|", fontsize=11)
ax_a.grid(True, linestyle="--", alpha=0.6)
ax_a.legend(fontsize=10, loc="upper left")

# -------------------------------------------------------------------------
# Panel B: Relative Error vs Force
# -------------------------------------------------------------------------
ax_b = axes[0, 1]
for N in Ns:
    data = results[N]
    err_pct = [abs(v - (f / 16.0)) / (f / 16.0) * 100.0 for f, v in zip(data["forces"], data["velocities"])]
    ax_b.plot(data["forces"], err_pct, f"-{markers[N]}", color=colors[N],
              linewidth=2.0, markersize=8, label=f"N = {N} (Mean Err: {np.mean(err_pct):.2f}%)")

ax_b.set_title("Relative Discretization Error vs Analytical Solution (%)", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Applied Force F", fontsize=11)
ax_b.set_ylabel("Error %: |V - V_exact| / V_exact × 100%", fontsize=11)
ax_b.grid(True, linestyle="--", alpha=0.6)
ax_b.legend(fontsize=10, loc="upper right")

# -------------------------------------------------------------------------
# Panel C: Effective Hydrodynamic Radius Rh vs N
# -------------------------------------------------------------------------
# Exact Stokes drag gives V = F / (16 eta R_eff) => R_eff = F / (16 eta V)
ax_c = axes[1, 0]
r_effs = []
for N in Ns:
    data = results[N]
    # compute average R_eff across forces
    reff_vals = [f / (16.0 * 1.0 * v) for f, v in zip(data["forces"], data["velocities"])]
    r_effs.append(np.mean(reff_vals))

ax_c.plot(Ns, r_effs, "bo-", linewidth=2.2, markersize=8, label="Numerical R_eff")
ax_c.axhline(1.0, color="crimson", linestyle="--", linewidth=2.0, label="Geometric Radius R = 1.0")
for N, r_eff in zip(Ns, r_effs):
    ax_c.annotate(f"N={N}\nRh={r_eff:.3f}", xy=(N, r_eff), xytext=(N, r_eff + 0.04),
                  ha="center", fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

ax_c.set_ylim(0.9, 1.35)
ax_c.set_title("Effective Hydrodynamic Radius Rh Convergence", fontsize=12, fontweight="bold")
ax_c.set_xlabel("Discretization Resolution N", fontsize=11)
ax_c.set_ylabel("Effective Hydrodynamic Radius R_eff", fontsize=11)
ax_c.grid(True, linestyle="--", alpha=0.6)
ax_c.legend(fontsize=10, loc="upper right")

# -------------------------------------------------------------------------
# Panel D: Visual Discretization of the 4 Disc Meshes
# -------------------------------------------------------------------------
ax_d = axes[1, 1]
# Draw 4 mini sub-discs in a 2x2 grid inside Panel D
sub_centers = {
    19: (-0.6, 0.6),
    37: (0.6, 0.6),
    61: (-0.6, -0.6),
    91: (0.6, -0.6)
}
scale = 0.45

for N in Ns:
    cx, cy = sub_centers[N]
    v_file = os.path.join(struct_dir, f"disc_N_{N}.vertex")
    with open(v_file, "r") as f:
        pts = np.array([[float(p) for p in l.split()] for l in f.readlines()[1:] if l.strip()])
    
    rad_b = results[N]["a"] * scale
    # Outer circle
    ax_d.add_patch(patches.Circle((cx, cy), 1.0 * scale, fill=False, edgecolor="black", linewidth=1.5))
    for pt in pts:
        bx = cx + pt[0] * scale
        by = cy + pt[1] * scale
        circ = patches.Circle((bx, by), rad_b, color=colors[N], alpha=0.35, edgecolor="navy", linewidth=0.5)
        ax_d.add_patch(circ)
        ax_d.plot(bx, by, "k.", markersize=2)
    ax_d.text(cx, cy - 1.25 * scale, f"N = {N} (m={rings[N]}, a={results[N]['a']:.3f})\nRh = {r_effs[Ns.index(N)]:.3f}",
              ha="center", va="top", fontsize=9, fontweight="bold")

ax_d.set_xlim(-1.25, 1.25)
ax_d.set_ylim(-1.35, 1.15)
ax_d.set_aspect("equal")
ax_d.set_title("Geometric Discretization Meshes (Overlapping Blobs)", fontsize=12, fontweight="bold")
ax_d.axis("off")

plt.tight_layout()

# Save
p1 = os.path.join(disc_dir, "disc_velocity_vs_force.png")
p2 = os.path.join(plots_dir, "disc_velocity_vs_force.png")
p3 = os.path.join(thesis_fig_dir, "disc_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.savefig(p3, dpi=300, bbox_inches="tight")
plt.close()

print(f"Generated and saved Disc Velocity vs Force plots:")
print(f"  -> {p1}")
print(f"  -> {p3}")

# Print summary table
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY TABLE:")
print(f"{'N':<6} {'Rings m':<10} {'Blob Radius a':<15} {'V(F=1000)':<12} {'Exact V':<10} {'Error %':<10} {'R_eff':<10}")
print("-" * 80)
for N in Ns:
    m = rings[N]
    a = results[N]["a"]
    v1000 = results[N]["velocities"][0]
    vex = 1000.0 / 16.0
    err = abs(v1000 - vex) / vex * 100.0
    reff = 1000.0 / (16.0 * v1000)
    print(f"{N:<6} {m:<10} {a:<15.4f} {v1000:<12.2f} {vex:<10.2f} {err:<10.2f}% {reff:<10.3f}")
print("=" * 80)
