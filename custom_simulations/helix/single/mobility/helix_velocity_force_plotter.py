import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

parser = argparse.ArgumentParser(description="Helix Mobility Scheme benchmark across N=19, 38, 75")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

Ns = [19, 38, 75]
blob_radii = {
    19: 0.2534,
    38: 0.1254,
    75: 0.0630
}
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = get_base_dir()
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
helix_dir = os.path.join(base_dir, "custom_simulations", "helix", "single")
mob_dir = os.path.join(helix_dir, "mobility")
struct_dir = os.path.join(helix_dir, "structures")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(thesis_fig_dir, exist_ok=True)
os.makedirs(mob_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

colors = {19: "#1f77b4", 38: "#ff7f0e", 75: "#2ca02c"}
markers = {19: "o", 38: "s", 75: "^"}

results = {}

print("=" * 80)
print("HELIX RESOLUTION BENCHMARK: MOBILITY SCHEME ACROSS N in [19, 38, 75]")
print(f"Testing Force values: {F_values}")
print("=" * 80)

for N in Ns:
    n_dir = os.path.join(mob_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    v_src = os.path.join(struct_dir, f"helix_N_{N}.vertex")
    v_dst = os.path.join(n_dir, f"helix_N_{N}.vertex")
    if not os.path.exists(v_dst) or args.rerun:
        import shutil
        shutil.copyfile(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    brad = blob_radii[N]
    forces, vels, omegas = [], [], []
    
    for F in F_values:
        g = F / N
        out_name = f"data/helix_N_{N}_F_{int(F)}"
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
structure helix_N_{N}.vertex single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Mobility] Running Helix N={N}, F={F}...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
            
        vel_data = np.loadtxt(vel_file)
        vz = abs(vel_data[2])
        wz = abs(vel_data[5])
        forces.append(F)
        vels.append(vz)
        omegas.append(wz)
        
    results[N] = {
        "forces": forces,
        "velocities": vels,
        "omegas": omegas,
        "mu_TT": vels[0] / forces[0],
        "mu_TR": omegas[0] / forces[0],
        "brad": brad
    }
    print(f"N={N:2d}: mu_TT = {results[N]['mu_TT']:.6f}, mu_TR (Chiral) = {results[N]['mu_TR']:.6f}")

print("\nAll mobility benchmark runs completed successfully!")
print("Generating publication-grade 3-panel figure...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Linear Velocity vs Force
ax1 = axes[0]
for N in Ns:
    ax1.plot(results[N]["forces"], results[N]["velocities"], 
             marker=markers[N], color=colors[N], linewidth=2.0, markersize=8,
             label=f"N={N} (a={results[N]['brad']:.4f})")
ax1.set_title("Translational Velocity $V_z$ vs Force $F$", fontsize=13, fontweight="bold")
ax1.set_xlabel("Gravitational Force $F$ (units)", fontsize=11)
ax1.set_ylabel("Terminal Velocity $|V_z|$ (units/s)", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=10)

# Panel 2: Chiral Angular Velocity vs Force
ax2 = axes[1]
for N in Ns:
    ax2.plot(results[N]["forces"], results[N]["omegas"], 
             marker=markers[N], color=colors[N], linewidth=2.0, markersize=8,
             label=f"N={N} ($\\mu_{{TR}}={results[N]['mu_TR']:.5f}$)")
ax2.set_title("Chiral Angular Velocity $|\Omega_z|$ vs Force $F$", fontsize=13, fontweight="bold")
ax2.set_xlabel("Gravitational Force $F$ (units)", fontsize=11)
ax2.set_ylabel("Induced Spin $|\Omega_z|$ (rad/s)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(fontsize=10)

# Panel 3: Mobility Scaling across Discretization N
ax3 = axes[2]
mu_tt_vals = [results[N]["mu_TT"] for N in Ns]
mu_tr_vals = [results[N]["mu_TR"] for N in Ns]

ax3_twin = ax3.twinx()
line1 = ax3.plot(Ns, mu_tt_vals, "bo-", linewidth=2.0, markersize=8, label="$\mu_{TT}$ (Translational)")
line2 = ax3_twin.plot(Ns, mu_tr_vals, "rs--", linewidth=2.0, markersize=8, label="$\mu_{TR}$ (Chiral Spin)")

ax3.set_title("Mobility Convergence across Discretization $N$", fontsize=13, fontweight="bold")
ax3.set_xlabel("Number of Blobs $N$", fontsize=11)
ax3.set_ylabel("Translational Mobility $\mu_{TT} = V_z / F$", color="blue", fontsize=11)
ax3_twin.set_ylabel("Chiral Coupling $\mu_{TR} = \Omega_z / F$", color="red", fontsize=11)
ax3.grid(True, linestyle="--", alpha=0.6)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="center right", fontsize=10)

plt.tight_layout()

p1 = os.path.join(mob_dir, "helix_velocity_vs_force.png")
p2 = os.path.join(thesis_fig_dir, "helix_velocity_vs_force.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")

print("\n" + "=" * 80)
print(f"{'N':<6} {'Blob Radius a':<15} {'Trans Mobility mu_TT':<22} {'Chiral Coupling mu_TR':<22}")
print("-" * 80)
for N in Ns:
    print(f"{N:<6} {results[N]['brad']:<15.4f} {results[N]['mu_TT']:<22.6f} {results[N]['mu_TR']:<22.6f}")
print("=" * 80)
