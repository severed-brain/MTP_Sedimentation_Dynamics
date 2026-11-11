import os
import sys
import math
import csv
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"d:\sedimentation_dynamics"

base_dir = get_base_dir()
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
struct_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
data_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "data")
plots_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
scratch_dir = os.path.join(base_dir, "dump", "rotational_scratch")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)
os.makedirs(scratch_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

def run_mobility_solve(v_name, brad, N, F, d_r, q1=(1,0,0,0), q2=(1,0,0,0), tag="rot_pair"):
    v_src = os.path.join(struct_dir, v_name)
    v_dst = os.path.join(scratch_dir, v_name)
    if not os.path.exists(v_dst):
        shutil.copyfile(v_src, v_dst)
        
    R = 1.0
    d = d_r * R
    x1 = -d / 2.0
    x2 = +d / 2.0
    
    clones_path = os.path.join(scratch_dir, f"{tag}.clones")
    with open(clones_path, "w") as f:
        f.write("2\n")
        f.write(f"{x1:.6f} 0.0 0.0 {q1[0]:.6f} {q1[1]:.6f} {q1[2]:.6f} {q1[3]:.6f}\n")
        f.write(f"{x2:.6f} 0.0 0.0 {q2[0]:.6f} {q2[1]:.6f} {q2[2]:.6f} {q2[3]:.6f}\n")
        
    g_val = F / N
    inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {g_val:.8f}
blob_radius {brad:.6f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name {tag}
structure {v_name} {os.path.basename(clones_path)}
"""
    inp_path = os.path.join(scratch_dir, f"inp_{tag}.dat")
    with open(inp_path, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_path)]
    subprocess.run(cmd, cwd=scratch_dir, env=env, check=True, capture_output=True)
    
    vel_path = os.path.join(scratch_dir, f"{tag}.velocity.dat")
    vel = np.loadtxt(vel_path)
    
    for p in [clones_path, inp_path, vel_path]:
        if os.path.exists(p):
            os.remove(p)
            
    b1_vel = {
        "Vx": float(vel[0, 0]), "Vy": float(vel[0, 1]), "Vz": float(vel[0, 2]),
        "Wx": float(vel[0, 3]), "Wy": float(vel[0, 4]), "Wz": float(vel[0, 5])
    }
    b2_vel = {
        "Vx": float(vel[1, 0]), "Vy": float(vel[1, 1]), "Vz": float(vel[1, 2]),
        "Wx": float(vel[1, 3]), "Wy": float(vel[1, 4]), "Wz": float(vel[1, 5])
    }
    return b1_vel, b2_vel

def run_single_mobility(v_name, brad, N, F, q=(1,0,0,0), tag="single"):
    v_src = os.path.join(struct_dir, v_name)
    v_dst = os.path.join(scratch_dir, v_name)
    if not os.path.exists(v_dst):
        shutil.copyfile(v_src, v_dst)
        
    clones_path = os.path.join(scratch_dir, f"{tag}.clones")
    with open(clones_path, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        
    g_val = F / N
    inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {g_val:.8f}
blob_radius {brad:.6f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name {tag}
structure {v_name} {os.path.basename(clones_path)}
"""
    inp_path = os.path.join(scratch_dir, f"inp_{tag}.dat")
    with open(inp_path, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_path)]
    subprocess.run(cmd, cwd=scratch_dir, env=env, check=True, capture_output=True)
    
    vel_path = os.path.join(scratch_dir, f"{tag}.velocity.dat")
    vel = np.loadtxt(vel_path)
    
    for p in [clones_path, inp_path, vel_path]:
        if os.path.exists(p):
            os.remove(p)
            
    return {
        "Vx": float(vel[0]), "Vy": float(vel[1]), "Vz": float(vel[2]),
        "Wx": float(vel[3]), "Wy": float(vel[4]), "Wz": float(vel[5])
    }

def main():
    print("=" * 95)
    print("COMPREHENSIVE MULTI-BODY ROTATIONAL DYNAMICS AND HYDRODYNAMIC ROLLING SUITE")
    print("Measuring Induced Angular Velocities (Wx, Wy, Wz) Across Geometries and Separations d/R")
    print("=" * 95)
    
    d_list = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    
    # 1. Load Sphere high-res data or recompute for d_list
    sphere_rot = []
    print("\n[PART 1] Sphere Pair Induced Counter-Rotation (N=162)...")
    for d_r in d_list:
        b1, b2 = run_mobility_solve("sphere_N_162.vertex", 0.1380, 162, 1000.0, d_r, tag=f"sph_{d_r}")
        sphere_rot.append({"d_r": d_r, "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"], "Vz": abs(b1["Vz"])})
        print(f"  d/R = {d_r:4.1f}: Wy_b1 = {b1['Wy']:+7.4f} rad/s, Wy_b2 = {b2['Wy']:+7.4f} rad/s, Vz = {abs(b1['Vz']):.2f}")
        
    # 2. Ellipsoid prolate (theta = 0 deg and theta = 90 deg)
    print("\n[PART 2] Ellipsoid Pair Rotations (N=162)...")
    ell_rot_0 = []
    ell_rot_90 = []
    # theta = 0: align along z (no rotation quaternion)
    # theta = 90: rotated 90 deg around y: q = (cos(45), 0, sin(45), 0)
    q90 = (math.cos(math.pi/4), 0.0, math.sin(math.pi/4), 0.0)
    for d_r in d_list:
        b1_0, b2_0 = run_mobility_solve("ellipsoid_prolate_N_162.vertex", 0.1393, 162, 1000.0, d_r, tag=f"ell0_{d_r}")
        ell_rot_0.append({"d_r": d_r, "Wy_b1": b1_0["Wy"], "Wy_b2": b2_0["Wy"]})
        
        b1_90, b2_90 = run_mobility_solve("ellipsoid_prolate_N_162.vertex", 0.1393, 162, 1000.0, d_r, q1=q90, q2=q90, tag=f"ell90_{d_r}")
        ell_rot_90.append({"d_r": d_r, "Wy_b1": b1_90["Wy"], "Wy_b2": b2_90["Wy"]})
        print(f"  d/R = {d_r:4.1f}: θ=0° Wy={b1_0['Wy']:+7.4f} | θ=90° Wy={b1_90['Wy']:+7.4f}")
        
    # 3. Disc (N=91) broadside
    print("\n[PART 3] Disc Pair Rotations (N=91)...")
    disc_rot = []
    for d_r in d_list:
        b1, b2 = run_mobility_solve("disc_N_91.vertex", 0.1000, 91, 1000.0, d_r, tag=f"dsc_{d_r}")
        disc_rot.append({"d_r": d_r, "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"]})
        print(f"  d/R = {d_r:4.1f}: Wy_b1 = {b1['Wy']:+7.4f} rad/s")
        
    # 4. Cylinder (N=86)
    print("\n[PART 4] Cylinder Pair Rotations (N=86)...")
    cyl_rot = []
    for d_r in d_list:
        b1, b2 = run_mobility_solve("cylinder_N_86.vertex", 0.1474, 86, 1000.0, d_r, tag=f"cyl_{d_r}")
        cyl_rot.append({"d_r": d_r, "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"]})
        print(f"  d/R = {d_r:4.1f}: Wy_b1 = {b1['Wy']:+7.4f} rad/s")
        
    # 5. Cube (N=56)
    print("\n[PART 5] Cube Pair Rotations (N=56)...")
    cube_rot = []
    for d_r in d_list:
        b1, b2 = run_mobility_solve("cube_N_56.vertex", 0.3333, 56, 1000.0, d_r, tag=f"cube_{d_r}")
        cube_rot.append({"d_r": d_r, "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"]})
        print(f"  d/R = {d_r:4.1f}: Wy_b1 = {b1['Wy']:+7.4f} rad/s")
        
    # 6. Chiral Particles: Helix (N=38) and Boomerang (N=15) Auto-Rotation Wz
    print("\n[PART 6] Chiral Particle Spin and Coupling (Helix & Boomerang)...")
    helix_single = run_single_mobility("helix_N_38.vertex", 0.1254, 38, 1000.0, tag="helix_single")
    wz_helix_single = helix_single["Wz"]
    print(f"  Single Helix Chiral Spin: Wz = {wz_helix_single:+.4f} rad/s")
    
    helix_rot = []
    for d_r in d_list:
        b1, b2 = run_mobility_solve("helix_N_38.vertex", 0.1254, 38, 1000.0, d_r, tag=f"hlx_{d_r}")
        helix_rot.append({
            "d_r": d_r, "Wz_b1": b1["Wz"], "Wz_b2": b2["Wz"], "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"]
        })
        print(f"  d/R = {d_r:4.1f}: Wz_b1 = {b1['Wz']:+7.4f} rad/s, Wy_b1 = {b1['Wy']:+7.4f} rad/s")
        
    # Save CSV
    summary_csv = os.path.join(data_dir, "rotational_velocities_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_r", "sphere_Wy", "ellipsoid_0_Wy", "ellipsoid_90_Wy", "disc_Wy", "cylinder_Wy", "cube_Wy", "helix_Wz", "helix_Wy"])
        for i, d_r in enumerate(d_list):
            writer.writerow([
                d_r,
                sphere_rot[i]["Wy_b1"],
                ell_rot_0[i]["Wy_b1"],
                ell_rot_90[i]["Wy_b1"],
                disc_rot[i]["Wy_b1"],
                cyl_rot[i]["Wy_b1"],
                cube_rot[i]["Wy_b1"],
                helix_rot[i]["Wz_b1"],
                helix_rot[i]["Wy_b1"]
            ])
    print(f"\nSaved rotational velocities data to: {summary_csv}")
    
    # -------------------------------------------------------------------------
    # GENERATE 4-PANEL PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    print("\nGenerating publication figure for rotational dynamics...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: Induced Counter-Rotation Wy vs d/R across Geometries
    ax_a = axes[0, 0]
    ax_a.plot(d_list, [r["Wy_b1"] for r in sphere_rot], 'ro-', linewidth=2.2, markersize=8, label="Sphere (N=162)")
    ax_a.plot(d_list, [r["Wy_b1"] for r in ell_rot_0], 's-', color="#1f77b4", linewidth=2.0, markersize=7, label="Prolate Ellipsoid θ=0°")
    ax_a.plot(d_list, [r["Wy_b1"] for r in ell_rot_90], 'v-', color="#2ca02c", linewidth=2.0, markersize=7, label="Prolate Ellipsoid θ=90°")
    ax_a.plot(d_list, [r["Wy_b1"] for r in disc_rot], 'p-', color="#9467bd", linewidth=2.0, markersize=7, label="Disc (Broadside)")
    ax_a.plot(d_list, [r["Wy_b1"] for r in cyl_rot], 'd-', color="#8c564b", linewidth=2.0, markersize=7, label="Cylinder (N=86)")
    ax_a.plot(d_list, [r["Wy_b1"] for r in cube_rot], 'h-', color="#ff7f0e", linewidth=2.0, markersize=7, label="Cube (N=56)")
    
    ax_a.axhline(0.0, color='gray', linestyle=':', alpha=0.7)
    ax_a.set_title(r"(A) Induced Counter-Rotation Angular Velocity $\Omega_y$ vs $d/R$", fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_a.set_ylabel(r"Induced Rolling Rate $\Omega_y$ (rad/s)", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=9)
    
    # Panel B: Power-Law Asymptotic Decay (Log-Log Plot: ~d^-2 vs ~d^-1)
    ax_b = axes[0, 1]
    ds = np.array(d_list)
    ax_b.loglog(ds, [abs(r["Wy_b1"]) for r in sphere_rot], 'ro-', linewidth=2.2, markersize=8, label=r'Sphere $\Omega_y$ (Simulated)')
    ax_b.loglog(ds, [abs(r["Wy_b1"]) for r in disc_rot], 'p-', color="#9467bd", linewidth=2.0, markersize=7, label=r'Disc $\Omega_y$ (Simulated)')
    ax_b.loglog(ds, [abs(r["Wy_b1"]) for r in cube_rot], 'h-', color="#ff7f0e", linewidth=2.0, markersize=7, label=r'Cube $\Omega_y$ (Simulated)')
    
    # Reference slopes
    d_dense = np.logspace(np.log10(1.8), np.log10(11.0), 100)
    ref_rot = 22.0 * (d_dense**(-2.0))
    ref_trans = 0.75 * (d_dense**(-1.0)) * 50.0
    ax_b.loglog(d_dense, ref_rot, 'k--', linewidth=2.0, label=r'Theoretical Dipole Scaling $\sim (R/d)^2$')
    ax_b.loglog(d_dense, ref_trans, 'k:', linewidth=1.8, label=r'Translational Monopole Scaling $\sim (R/d)^1$')
    
    ax_b.set_title(r"(B) Asymptotic Decay: Dipole $\sim d^{-2}$ vs Monopole $\sim d^{-1}$", fontsize=12, fontweight='bold')
    ax_b.set_xlabel(r"Separation Distance Ratio $d/R$ (Log Scale)", fontsize=11)
    ax_b.set_ylabel(r"Angular Velocity Magnitude $|\Omega_y|$ (rad/s)", fontsize=11)
    ax_b.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=9)
    
    # Panel C: Asymmetric Equal-and-Opposite Rolling Verification (Body 1 vs Body 2)
    ax_c = axes[1, 0]
    ax_c.plot(d_list, [r["Wy_b1"] for r in sphere_rot], 'r^-', linewidth=2.2, markersize=8, label=r"Sphere B1 ($x < 0$): $\Omega_y > 0$ (CW)")
    ax_c.plot(d_list, [r["Wy_b2"] for r in sphere_rot], 'bv--', linewidth=2.2, markersize=8, label=r"Sphere B2 ($x > 0$): $\Omega_y < 0$ (CCW)")
    ax_c.plot(d_list, [r["Wy_b1"] for r in cube_rot], 's-', color="#ff7f0e", linewidth=1.8, markersize=6, label=r"Cube B1 ($x < 0$)")
    ax_c.plot(d_list, [r["Wy_b2"] for r in cube_rot], 's--', color="#ff7f0e", linewidth=1.8, markersize=6, label=r"Cube B2 ($x > 0$)")
    
    ax_c.axhline(0.0, color='black', linewidth=1.0)
    ax_c.set_title(r"(C) Hydrodynamic Gear-Mesh Rolling Symmetry ($\Omega_y^{(1)} = -\Omega_y^{(2)}$)", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_c.set_ylabel(r"Angular Velocity $\Omega_y$ (rad/s)", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=9)
    
    # Panel D: Chiral Auto-Rotation Wz and Translation-Rotation Coupling
    ax_d = axes[1, 1]
    wz_pair = [r["Wz_b1"] for r in helix_rot]
    wy_pair = [r["Wy_b1"] for r in helix_rot]
    ax_d.plot(d_list, wz_pair, 'mo-', linewidth=2.2, markersize=8, label=r"Helix Intrinsic Spin $\Omega_z$ in Pair")
    ax_d.axhline(wz_helix_single, color='m', linestyle='--', linewidth=1.8, label=f"Solitary Helix Spin $\Omega_0 = {wz_helix_single:.2f}$ rad/s")
    ax_d.plot(d_list, wy_pair, 'co-', linewidth=2.0, markersize=7, label=r"Helix Induced Rolling $\Omega_y$ in Pair")
    ax_d.axhline(0.0, color='gray', linestyle=':', alpha=0.7)
    
    ax_d.set_title(r"(D) Chiral Hydrodynamic Coupling: Spin $\Omega_z$ and Rolling $\Omega_y$", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_d.set_ylabel("Angular Velocity (rad/s)", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "multi_body_rotational_dynamics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "multi_body_rotational_dynamics.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'multi_body_rotational_dynamics.png')}")
    print("=" * 95)
    print("ROTATIONAL DYNAMICS SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()

