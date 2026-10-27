import os
import sys
import math
import shutil
import subprocess
import numpy as np
import pandas as pd
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
knot_dir = os.path.join(base_dir, "custom_simulations", "trefoil_knot")
struct_dir = os.path.join(knot_dir, "structures")
mob_dir = os.path.join(knot_dir, "mobility")
data_dir = os.path.join(mob_dir, "data")
plots_dir = os.path.join(mob_dir, "plots")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

def run_mobility_solve(vertex_file, blob_radius, F_total, angle_deg=0.0, output_prefix="trefoil_mob"):
    """Run RigidMultiblobsWall Stokesian mobility solve"""
    with open(vertex_file, "r") as f:
        data_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        N = int(data_lines[0].split()[0])
        
    g_val = F_total / N
    
    # Quaternion for pitch angle theta around Y axis
    ang_rad = math.radians(angle_deg)
    qw = math.cos(ang_rad / 2.0)
    qy = math.sin(ang_rad / 2.0)
    
    clones_path = os.path.join(mob_dir, f"{output_prefix}.clones")
    with open(clones_path, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {qw:.8f} 0.0 {qy:.8f} 0.0\n")
        
    v_target = os.path.join(mob_dir, os.path.basename(vertex_file))
    if not os.path.exists(v_target):
        shutil.copyfile(vertex_file, v_target)
        
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
    inp_file = os.path.join(mob_dir, f"input_{output_prefix}.dat")
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_file)]
    subprocess.run(cmd, cwd=mob_dir, env=env, check=True, capture_output=True)
    
    vel_file = os.path.join(mob_dir, f"{output_prefix}.velocity.dat")
    vel = np.loadtxt(vel_file)
    
    # Clean up intermediate run files
    for p in [inp_file, clones_path, vel_file]:
        if os.path.exists(p):
            os.remove(p)
            
    return {
        "Vx": float(vel[0]), "Vy": float(vel[1]), "Vz": float(vel[2]),
        "Wx": float(vel[3]), "Wy": float(vel[4]), "Wz": float(vel[5])
    }

def main():
    print("=" * 80)
    print("CHIRAL TREFOIL KNOT MOBILITY & SEDIMENTATION BENCHMARKS (N=64)")
    print("=" * 80)
    
    vertex_file = os.path.join(struct_dir, "trefoil_knot_N_64.vertex")
    blob_radius = 0.1288
    N = 64
    
    # 1. Force Sweep (Stokes Linearity Test at theta = 0 deg)
    print("\n[PART 1] Running Force Sweep (F = 1000 to 5000 units)...")
    forces = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    force_records = []
    
    for F in forces:
        res = run_mobility_solve(vertex_file, blob_radius, F, angle_deg=0.0, output_prefix=f"mob_F_{int(F)}")
        force_records.append({
            "N": N, "blob_radius": blob_radius, "Force": F,
            "Vz": abs(res["Vz"]), "Vx": abs(res["Vx"]), "Wz": abs(res["Wz"])
        })
        print(f"  F = {F:6.0f} -> Vz = {abs(res['Vz']):8.4f} u/s, Vx = {abs(res['Vx']):8.4f} u/s, Wz = {abs(res['Wz']):8.4f} rad/s")
        
    df_force = pd.DataFrame(force_records)
    csv_force = os.path.join(data_dir, "trefoil_knot_mobility.csv")
    df_force.to_csv(csv_force, index=False)
    print(f"Saved: {csv_force}")
    
    # 2. Angle Sweep (Hydrodynamic Drift & Anisotropy Test at F = 1000 units)
    print("\n[PART 2] Running Orientation Angle Sweep (theta = 0 to 90 deg)...")
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    angle_records = []
    
    for ang in angles:
        res = run_mobility_solve(vertex_file, blob_radius, 1000.0, angle_deg=ang, output_prefix=f"angle_{int(ang)}")
        angle_records.append({
            "N": N, "blob_radius": blob_radius, "Force": 1000.0, "angle_deg": ang,
            "Vz": abs(res["Vz"]), "Vx": abs(res["Vx"]), "Wz": abs(res["Wz"])
        })
        print(f"  theta = {ang:4.1f} deg -> Vz = {abs(res['Vz']):8.4f} u/s, Vx = {abs(res['Vx']):8.4f} u/s, Wz = {abs(res['Wz']):8.4f} rad/s")
        
    df_angle = pd.DataFrame(angle_records)
    csv_angle = os.path.join(data_dir, "trefoil_knot_angle_sweep.csv")
    df_angle.to_csv(csv_angle, index=False)
    print(f"Saved: {csv_angle}")
    
    # Plotting mobility curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Velocity vs Force
    axes[0].plot(df_force["Force"], df_force["Vz"], "o-", color="#1f77b4", linewidth=2.0, markersize=7, label="Vz (Settling)")
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(df_force["Force"], df_force["Wz"], "s--", color="#e377c2", linewidth=2.0, markersize=7, label="Wz (Chiral Spin)")
    axes[0].set_title("Trefoil Knot Velocity & Chiral Spin vs Force (N=64)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Applied Force F", fontsize=10)
    axes[0].set_ylabel("Terminal Velocity Vz (units/s)", fontsize=10, color="#1f77b4")
    ax0_twin.set_ylabel("Chiral Angular Velocity Wz (rad/s)", fontsize=10, color="#e377c2")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Angle sweep
    axes[1].plot(df_angle["angle_deg"], df_angle["Vz"], "ro-", linewidth=2.0, label="Vz (Settling)")
    axes[1].plot(df_angle["angle_deg"], df_angle["Vx"], "bs--", linewidth=2.0, label="Vx (Lateral Drift)")
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(df_angle["angle_deg"], df_angle["Wz"], "m^:", linewidth=2.0, label="Wz (Chiral Spin)")
    axes[1].set_title("Sedimentation Velocities vs Pitch Angle (N=64, F=1000)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Pitch Angle theta (degrees)", fontsize=10)
    axes[1].set_ylabel("Translational Velocity (units/s)", fontsize=10)
    ax1_twin.set_ylabel("Angular Velocity Wz (rad/s)", fontsize=10, color="m")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8.5)
    
    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "trefoil_knot_simulation_benchmarks.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark plot: {plot_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
