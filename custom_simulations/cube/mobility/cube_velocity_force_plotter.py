import os
import sys
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
cube_dir = os.path.join(base_dir, "custom_simulations", "cube")
struct_dir = os.path.join(cube_dir, "structures")
mob_dir = os.path.join(cube_dir, "mobility")
data_dir = os.path.join(mob_dir, "data")
plots_dir = os.path.join(mob_dir, "plots")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

def run_mobility_solve(vertex_file, blob_radius, F_total, output_prefix):
    """Run RigidMultiblobsWall mobility solve for a given force"""
    # Read number of blobs N
    with open(vertex_file, "r") as f:
        N = int(f.readline().strip())
        
    g_val = F_total / N
    
    # Create clones file (1 body at origin, upright orientation: quaternion [1, 0, 0, 0])
    clones_path = os.path.join(mob_dir, "cube_upright.clones")
    with open(clones_path, "w") as f:
        f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {g_val:.8f}
blob_radius {blob_radius:.6f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name {output_prefix}
structure {os.path.basename(vertex_file)} {os.path.basename(clones_path)}
"""
    inp_file = os.path.join(mob_dir, f"input_{os.path.basename(output_prefix)}.dat")
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    # Copy vertex file to mob_dir so relative path in input works
    import shutil
    v_target = os.path.join(mob_dir, os.path.basename(vertex_file))
    if not os.path.exists(v_target):
        shutil.copyfile(vertex_file, v_target)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_file)]
    subprocess.run(cmd, cwd=mob_dir, env=env, check=True, capture_output=True)
    
    # Read output velocity
    vel_file = os.path.join(mob_dir, f"{output_prefix}.velocity.dat")
    vel = np.loadtxt(vel_file)
    Vx, Vy, Vz = float(vel[0]), float(vel[1]), float(vel[2])
    Wx, Wy, Wz = float(vel[3]), float(vel[4]), float(vel[5])
    return {
        "Vx": abs(Vx), "Vy": abs(Vy), "Vz": abs(Vz),
        "Wx": abs(Wx), "Wy": abs(Wy), "Wz": abs(Wz)
    }

def main():
    print("=" * 80)
    print("RIGID MULTIBLOB STOKESIAN SIMULATIONS: CUBE MOBILITY VS FORCE")
    print("=" * 80)
    
    resolutions = [
        {"N": 56, "file": os.path.join(struct_dir, "cube_N_56.vertex"), "a": 0.3333},
        {"N": 98, "file": os.path.join(struct_dir, "cube_N_98.vertex"), "a": 0.2500},
        {"N": 152, "file": os.path.join(struct_dir, "cube_N_152.vertex"), "a": 0.2000}
    ]
    
    forces = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    results = {}
    
    for r in resolutions:
        N = r["N"]
        results[N] = []
        print(f"\nSimulating Cube N = {N} (blob radius a = {r['a']:.4f}):")
        print(f"{'Force (F)':<12} {'Vz (units/s)':<16} {'Vx (units/s)':<16} {'Mobility (Vz/F)':<18}")
        print("-" * 62)
        
        for F in forces:
            prefix = f"data/cube_N_{N}_F_{int(F)}"
            res = run_mobility_solve(r["file"], r["a"], F, prefix)
            mob = res["Vz"] / F
            results[N].append({
                "F": F, "Vz": res["Vz"], "Vx": res["Vx"], "Wz": res["Wz"], "mob": mob
            })
            print(f"{F:<12.1f} {res['Vz']:<16.4f} {res['Vx']:<16.4e} {mob:<18.6f}")
            
    # Save results to data table
    summary_csv = os.path.join(data_dir, "cube_mobility_stokes_law.csv")
    with open(summary_csv, "w") as f:
        f.write("N,blob_radius,Force,Vz,Vx,Wz,mobility\n")
        for r in resolutions:
            N = r["N"]
            for row in results[N]:
                f.write(f"{N},{r['a']},{row['F']},{row['Vz']},{row['Vx']},{row['Wz']},{row['mob']}\n")
    print(f"\nSaved mobility summary table: {summary_csv}")
    
    # Plot Velocity vs Force
    plt.figure(figsize=(9, 6))
    colors = {56: "#1f77b4", 98: "#2ca02c", 152: "#d62728"}
    markers = {56: "o", 98: "s", 152: "^"}
    
    for r in resolutions:
        N = r["N"]
        F_vals = [row["F"] for row in results[N]]
        V_vals = [row["Vz"] for row in results[N]]
        mobs = [row["mob"] for row in results[N]]
        mean_mob = np.mean(mobs)
        plt.plot(F_vals, V_vals, marker=markers[N], color=colors[N], linewidth=2.0, markersize=8,
                 label=f"Cube $N={N}$ ($a={r['a']:.2f}$, $\mu={mean_mob:.5f}$)")
                 
    plt.title("Cube Terminal Sedimentation Velocity vs. Applied Force", fontsize=13, fontweight="bold")
    plt.xlabel("Total Gravitational Force $F$", fontsize=11)
    plt.ylabel("Vertical Sedimentation Velocity $V_z$ (units/s)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    fig_path = os.path.join(plots_dir, "cube_velocity_vs_force.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
