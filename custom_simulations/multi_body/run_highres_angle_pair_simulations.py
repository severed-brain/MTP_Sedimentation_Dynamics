import os
import sys
import math
import time
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
rmw_dir = os.path.join(base_dir, "RigidMultiblobsWall")
struct_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
data_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "data")
plots_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
scratch_dir = os.path.join(base_dir, "dump", "highres_angle_pairs_scratch")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)
os.makedirs(scratch_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{rmw_dir}"

def make_quaternion_y(ang_deg):
    rad = math.radians(ang_deg)
    return (math.cos(rad / 2.0), 0.0, math.sin(rad / 2.0), 0.0)

def run_mobility_pair(v_name, brad, N, F, d_r, ang1, ang2, tag="pair"):
    v_src = os.path.join(struct_dir, v_name)
    v_dst = os.path.join(scratch_dir, v_name)
    if not os.path.exists(v_dst):
        shutil.copyfile(v_src, v_dst)
        
    q1 = make_quaternion_y(ang1)
    q2 = make_quaternion_y(ang2)
    
    # Separation along X-axis: Body 1 at -d/2, Body 2 at +d/2
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
    t0 = time.time()
    subprocess.run(cmd, cwd=scratch_dir, env=env, check=True, capture_output=True)
    t_solve = time.time() - t0
    
    vel_path = os.path.join(scratch_dir, f"{tag}.velocity.dat")
    vel = np.loadtxt(vel_path)
    
    # Cleanup run files
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
    return b1_vel, b2_vel, t_solve

def run_single_body(v_name, brad, N, F, ang):
    v_src = os.path.join(struct_dir, v_name)
    v_dst = os.path.join(scratch_dir, v_name)
    if not os.path.exists(v_dst):
        shutil.copyfile(v_src, v_dst)
        
    q = make_quaternion_y(ang)
    clones_path = os.path.join(scratch_dir, f"single_{ang}.clones")
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
output_name single_{ang}
structure {v_name} {os.path.basename(clones_path)}
"""
    inp_path = os.path.join(scratch_dir, f"inp_single_{ang}.dat")
    with open(inp_path, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_path)]
    subprocess.run(cmd, cwd=scratch_dir, env=env, check=True, capture_output=True)
    
    vel_path = os.path.join(scratch_dir, f"single_{ang}.velocity.dat")
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
    print("HIGH-RESOLUTION MULTI-BODY SIMULATIONS: MISMATCHED & SYMMETRIC ANGLE PAIRS")
    print("===========================================================================================")
    
    d_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    # -------------------------------------------------------------------------
    # PART 1: HIGH-RES SPHERE (N=162)
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running High-Resolution Sphere Pair Simulations (N=162)...")
    v_sph = run_single_body("sphere_N_162.vertex", 0.1380, 162, 1000.0, 0.0)
    v0_sph = abs(v_sph["Vz"])
    print(f"  Single Sphere (N=162) Solitary V0 = {v0_sph:.4f} units/s")
    
    sph_records = []
    for d_r in d_list:
        b1, b2, t_s = run_mobility_pair("sphere_N_162.vertex", 0.1380, 162, 1000.0, d_r, 0.0, 0.0, f"sph_d_{int(d_r)}")
        vz_avg = (abs(b1["Vz"]) + abs(b2["Vz"])) / 2.0
        ratio = vz_avg / v0_sph
        stokeslet_th = 1.0 + 0.75 / d_r
        sph_records.append({
            "d_r": d_r, "N": 162, "V0": v0_sph, "Vz_b1": abs(b1["Vz"]), "Vz_b2": abs(b2["Vz"]),
            "Vz_avg": vz_avg, "ratio": ratio, "stokeslet_th": stokeslet_th,
            "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"], "solve_time_s": t_s
        })
        print(f"  d/R = {d_r:4.1f}: Vz = {vz_avg:7.4f} u/s, Ratio = {ratio:6.4f}x (Oseen = {stokeslet_th:.4f}x), Rot Wy = {b1['Wy']:+6.3f} rad/s")
        
    df_sph = pd.DataFrame(sph_records)
    csv_sph = os.path.join(data_dir, "highres_sphere_pairs.csv")
    df_sph.to_csv(csv_sph, index=False)
    print(f"Saved: {csv_sph}")
    
    # -------------------------------------------------------------------------
    # PART 2: HIGH-RES ELLIPSOID PAIRS (N=162) WITH MISMATCHED ANGLES
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running High-Resolution Ellipsoid Pair Simulations (N=162)...")
    ell_angles = [0.0, 45.0, 90.0]
    v0_ell = {}
    for ang in ell_angles:
        res = run_single_body("ellipsoid_prolate_N_162.vertex", 0.1393, 162, 1000.0, ang)
        v0_ell[ang] = {"Vz": abs(res["Vz"]), "Vx": abs(res["Vx"])}
        print(f"  Single Ellipsoid theta = {ang:4.1f} deg: Vz = {abs(res['Vz']):7.4f} u/s, Vx = {abs(res['Vx']):7.4f} u/s")
        
    # Mismatched and symmetric angle pairs (ang1, ang2)
    ell_angle_pairs = [
        (0.0, 45.0, "mismatched_0_45"),
        (45.0, 90.0, "mismatched_45_90"),
        (0.0, 90.0, "mismatched_0_90"),
        (0.0, 0.0, "symmetric_0_0"),
        (45.0, 45.0, "symmetric_45_45"),
        (90.0, 90.0, "symmetric_90_90")
    ]
    
    ell_records = []
    for ang1, ang2, tag in ell_angle_pairs:
        print(f"\n  -- Ellipsoid Angle Pair (theta1 = {ang1:.0f} deg, theta2 = {ang2:.0f} deg) --")
        for d_r in d_list:
            b1, b2, t_s = run_mobility_pair("ellipsoid_prolate_N_162.vertex", 0.1393, 162, 1000.0, d_r, ang1, ang2, f"ell_{tag}_d_{int(d_r)}")
            v1_z = abs(b1["Vz"])
            v2_z = abs(b2["Vz"])
            vz_avg = (v1_z + v2_z) / 2.0
            r1 = v1_z / v0_ell[ang1]["Vz"]
            r2 = v2_z / v0_ell[ang2]["Vz"]
            r_avg = vz_avg / ((v0_ell[ang1]["Vz"] + v0_ell[ang2]["Vz"]) / 2.0)
            
            ell_records.append({
                "ang1": ang1, "ang2": ang2, "pair_type": tag, "d_r": d_r,
                "V0_b1": v0_ell[ang1]["Vz"], "V0_b2": v0_ell[ang2]["Vz"],
                "Vz_b1": v1_z, "Vz_b2": v2_z, "Vz_avg": vz_avg, "delta_Vz": abs(v1_z - v2_z),
                "ratio_b1": r1, "ratio_b2": r2, "ratio_avg": r_avg,
                "Vx_b1": b1["Vx"], "Vx_b2": b2["Vx"],
                "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"],
                "solve_time_s": t_s
            })
            print(f"    d/R = {d_r:4.1f}: B1(Vz={v1_z:.2f}, Vx={b1['Vx']:+5.2f}, r={r1:.3f}x) | B2(Vz={v2_z:.2f}, Vx={b2['Vx']:+5.2f}, r={r2:.3f}x) | ΔVz = {abs(v1_z - v2_z):5.2f} u/s")
            
    df_ell = pd.DataFrame(ell_records)
    csv_ell = os.path.join(data_dir, "highres_angle_pairs_ellipsoid.csv")
    df_ell.to_csv(csv_ell, index=False)
    print(f"\nSaved: {csv_ell}")
    
    # -------------------------------------------------------------------------
    # PART 3: HIGH-RES DISC PAIRS (N=91) WITH MISMATCHED ANGLES
    # -------------------------------------------------------------------------
    print("\n[PART 3] Running High-Resolution Disc Pair Simulations (N=91)...")
    dsc_angles = [0.0, 45.0, 90.0]
    v0_dsc = {}
    for ang in dsc_angles:
        res = run_single_body("disc_N_91.vertex", 0.1000, 91, 1000.0, ang)
        v0_dsc[ang] = {"Vz": abs(res["Vz"]), "Vx": abs(res["Vx"])}
        print(f"  Single Disc theta = {ang:4.1f} deg: Vz = {abs(res['Vz']):7.4f} u/s, Vx = {abs(res['Vx']):7.4f} u/s")
        
    dsc_angle_pairs = [
        (0.0, 45.0, "mismatched_0_45"),
        (45.0, 90.0, "mismatched_45_90"),
        (0.0, 90.0, "mismatched_0_90"),
        (0.0, 0.0, "symmetric_0_0"),
        (45.0, 45.0, "symmetric_45_45"),
        (90.0, 90.0, "symmetric_90_90")
    ]
    
    dsc_records = []
    for ang1, ang2, tag in dsc_angle_pairs:
        print(f"\n  -- Disc Angle Pair (theta1 = {ang1:.0f} deg, theta2 = {ang2:.0f} deg) --")
        for d_r in d_list:
            b1, b2, t_s = run_mobility_pair("disc_N_91.vertex", 0.1000, 91, 1000.0, d_r, ang1, ang2, f"dsc_{tag}_d_{int(d_r)}")
            v1_z = abs(b1["Vz"])
            v2_z = abs(b2["Vz"])
            vz_avg = (v1_z + v2_z) / 2.0
            r1 = v1_z / v0_dsc[ang1]["Vz"]
            r2 = v2_z / v0_dsc[ang2]["Vz"]
            r_avg = vz_avg / ((v0_dsc[ang1]["Vz"] + v0_dsc[ang2]["Vz"]) / 2.0)
            
            dsc_records.append({
                "ang1": ang1, "ang2": ang2, "pair_type": tag, "d_r": d_r,
                "V0_b1": v0_dsc[ang1]["Vz"], "V0_b2": v0_dsc[ang2]["Vz"],
                "Vz_b1": v1_z, "Vz_b2": v2_z, "Vz_avg": vz_avg, "delta_Vz": abs(v1_z - v2_z),
                "ratio_b1": r1, "ratio_b2": r2, "ratio_avg": r_avg,
                "Vx_b1": b1["Vx"], "Vx_b2": b2["Vx"],
                "Wy_b1": b1["Wy"], "Wy_b2": b2["Wy"],
                "solve_time_s": t_s
            })
            print(f"    d/R = {d_r:4.1f}: B1(Vz={v1_z:.2f}, Vx={b1['Vx']:+5.2f}, r={r1:.3f}x) | B2(Vz={v2_z:.2f}, Vx={b2['Vx']:+5.2f}, r={r2:.3f}x) | ΔVz = {abs(v1_z - v2_z):5.2f} u/s")
            
    df_dsc = pd.DataFrame(dsc_records)
    csv_dsc = os.path.join(data_dir, "highres_angle_pairs_disc.csv")
    df_dsc.to_csv(csv_dsc, index=False)
    print(f"\nSaved: {csv_dsc}")
    
    # -------------------------------------------------------------------------
    # PART 4: GENERATE PUBLICATION-QUALITY PLOTS
    # -------------------------------------------------------------------------
    print("\nGenerating publication figures...")
    
    # FIGURE 1: ELLIPSOID ANGLE PAIRS (4 Panels)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: Mismatched Pairs Settling Velocity Vz vs d/R
    ax = axes[0, 0]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "θ₁=0°, θ₂=45°"),
                          ("mismatched_45_90", "#d62728", "θ₁=45°, θ₂=90°"),
                          ("mismatched_0_90", "#2ca02c", "θ₁=0°, θ₂=90°")]:
        sub = df_ell[df_ell["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["Vz_b1"], 'o-', color=col, linewidth=2.0, label=f"B1 ({lbl.split(',')[0].strip()})")
        ax.plot(sub["d_r"], sub["Vz_b2"], 's--', color=col, linewidth=1.8, alpha=0.8, label=f"B2 ({lbl.split(',')[1].strip()})")
    ax.set_title("Panel A: Asymmetric Angle Ellipsoid Settling ($V_z$ vs $d/R$)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Vertical Velocity $V_z$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=3)
    
    # Panel B: Differential Settling Velocity ΔVz = |Vz1 - Vz2|
    ax = axes[0, 1]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "Pair (0°, 45°)"),
                          ("mismatched_45_90", "#d62728", "Pair (45°, 90°)"),
                          ("mismatched_0_90", "#2ca02c", "Pair (0°, 90°)")]:
        sub = df_ell[df_ell["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["delta_Vz"], 'o-', color=col, linewidth=2.2, markersize=7, label=lbl)
    ax.set_title("Panel B: Hydrodynamic Velocity Splitting $\Delta V_z = |V_{z1} - V_{z2}|$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Speed Difference $\Delta V_z$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    
    # Panel C: Lateral Drift Velocities Vx (Oblique Hydrodynamic Repulsion/Attraction)
    ax = axes[1, 0]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "(0°, 45°)"),
                          ("mismatched_45_90", "#d62728", "(45°, 90°)"),
                          ("mismatched_0_90", "#2ca02c", "(0°, 90°)")]:
        sub = df_ell[df_ell["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["Vx_b1"], 'o-', color=col, linewidth=2.0, label=f"B1 {lbl}")
        ax.plot(sub["d_r"], sub["Vx_b2"], '^-', color=col, linewidth=2.0, label=f"B2 {lbl}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title("Panel C: Transverse Lateral Velocities $V_x$ vs $d/R$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Lateral Velocity $V_x$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=3)
    
    # Panel D: Drafting Enhancement Ratio V/V0 Decay
    ax = axes[1, 1]
    for tag, col, lbl in [("symmetric_0_0", "#1f77b4", "Sym (0°, 0°)"),
                          ("symmetric_45_45", "#d62728", "Sym (45°, 45°)"),
                          ("symmetric_90_90", "#2ca02c", "Sym (90°, 90°)"),
                          ("mismatched_0_45", "#9467bd", "Asym (0°, 45°)"),
                          ("mismatched_45_90", "#ff7f0e", "Asym (45°, 90°)")]:
        sub = df_ell[df_ell["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["ratio_avg"], 'o-', color=col, linewidth=2.0, markersize=6, label=lbl)
    ax.plot(d_list, [1.0 + 0.75/d for d in d_list], 'k--', linewidth=2.0, label="Oseen Stokeslet ($1 + 0.75/d$)")
    ax.set_title("Panel D: Cooperative Drafting Factor $V_{avg} / V_0$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Velocity Ratio $V / V_0$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    p_ell = os.path.join(plots_dir, "highres_ellipsoid_angle_pairs.png")
    p_ell_thesis = os.path.join(thesis_fig_dir, "highres_ellipsoid_angle_pairs.png")
    plt.savefig(p_ell, dpi=300, bbox_inches='tight')
    plt.savefig(p_ell_thesis, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_ell}")
    
    # FIGURE 2: DISC ANGLE PAIRS (4 Panels)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: Disc Settling Vz
    ax = axes[0, 0]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "θ₁=0°, θ₂=45°"),
                          ("mismatched_45_90", "#d62728", "θ₁=45°, θ₂=90°"),
                          ("mismatched_0_90", "#2ca02c", "θ₁=0°, θ₂=90°")]:
        sub = df_dsc[df_dsc["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["Vz_b1"], 'o-', color=col, linewidth=2.0, label=f"B1 ({lbl.split(',')[0].strip()})")
        ax.plot(sub["d_r"], sub["Vz_b2"], 's--', color=col, linewidth=1.8, alpha=0.8, label=f"B2 ({lbl.split(',')[1].strip()})")
    ax.set_title("Panel A: Asymmetric Angle Disc Settling ($V_z$ vs $d/R$)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Vertical Velocity $V_z$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=3)
    
    # Panel B: Disc ΔVz
    ax = axes[0, 1]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "Pair (0°, 45°)"),
                          ("mismatched_45_90", "#d62728", "Pair (45°, 90°)"),
                          ("mismatched_0_90", "#2ca02c", "Pair (0°, 90°)")]:
        sub = df_dsc[df_dsc["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["delta_Vz"], 'o-', color=col, linewidth=2.2, markersize=7, label=lbl)
    ax.set_title("Panel B: Disc Hydrodynamic Speed Difference $\Delta V_z$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Speed Difference $\Delta V_z$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=10)
    
    # Panel C: Disc Lateral Drift Vx
    ax = axes[1, 0]
    for tag, col, lbl in [("mismatched_0_45", "#1f77b4", "(0°, 45°)"),
                          ("mismatched_45_90", "#d62728", "(45°, 90°)"),
                          ("mismatched_0_90", "#2ca02c", "(0°, 90°)")]:
        sub = df_dsc[df_dsc["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["Vx_b1"], 'o-', color=col, linewidth=2.0, label=f"B1 {lbl}")
        ax.plot(sub["d_r"], sub["Vx_b2"], '^-', color=col, linewidth=2.0, label=f"B2 {lbl}")
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_title("Panel C: Disc Lateral Drift $V_x$ (Glide Coupling)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Lateral Velocity $V_x$ (units/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=3)
    
    # Panel D: Disc Enhancement Ratio
    ax = axes[1, 1]
    for tag, col, lbl in [("symmetric_0_0", "#1f77b4", "Sym (0°, 0°)"),
                          ("symmetric_45_45", "#d62728", "Sym (45°, 45°)"),
                          ("symmetric_90_90", "#2ca02c", "Sym (90°, 90°)"),
                          ("mismatched_0_45", "#9467bd", "Asym (0°, 45°)"),
                          ("mismatched_45_90", "#ff7f0e", "Asym (45°, 90°)")]:
        sub = df_dsc[df_dsc["pair_type"] == tag]
        ax.plot(sub["d_r"], sub["ratio_avg"], 'o-', color=col, linewidth=2.0, markersize=6, label=lbl)
    ax.plot(d_list, [1.0 + 0.75/d for d in d_list], 'k--', linewidth=2.0, label="Oseen Stokeslet ($1 + 0.75/d$)")
    ax.set_title("Panel D: Disc Cooperative Drafting Ratio $V_{avg} / V_0$", fontsize=12, fontweight='bold')
    ax.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax.set_ylabel("Velocity Ratio $V / V_0$", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    p_dsc = os.path.join(plots_dir, "highres_disc_angle_pairs.png")
    p_dsc_thesis = os.path.join(thesis_fig_dir, "highres_disc_angle_pairs.png")
    plt.savefig(p_dsc, dpi=300, bbox_inches='tight')
    plt.savefig(p_dsc_thesis, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_dsc}")
    
    # FIGURE 3: SPHERE PAIR SCALING (N=162)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(df_sph["d_r"], df_sph["Vz_avg"], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label="Numerical Stokesian ($N=162$)")
    ax1.axhline(v0_sph, color='red', linestyle='--', linewidth=2.0, label=f"Single Sphere ($V_0 = {v0_sph:.2f}$ u/s)")
    ax1.set_title("High-Resolution Sphere Pair ($N=162$): Velocity vs Separation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax1.set_ylabel("Settling Velocity $V_z$ (units/s)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(df_sph["d_r"], df_sph["ratio"], 'o-', color='#2ca02c', linewidth=2.5, markersize=8, label="Simulation Enhancement ($V/V_0$)")
    ax2.plot(df_sph["d_r"], df_sph["stokeslet_th"], 'r--', linewidth=2.0, label="Oseen Stokeslet Theory ($1 + 0.75/d$)")
    ax2.set_title("High-Resolution Sphere Pair: Drafting Enhancement Scaling", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax2.set_ylabel("Drafting Ratio $V / V_0$", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    p_sph = os.path.join(plots_dir, "highres_sphere_scaling.png")
    p_sph_thesis = os.path.join(thesis_fig_dir, "highres_sphere_scaling.png")
    plt.savefig(p_sph, dpi=300, bbox_inches='tight')
    plt.savefig(p_sph_thesis, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p_sph}")
    
    print("\n" + "=" * 95)
    print("ALL HIGH-RESOLUTION SIMULATIONS & FIGURES COMPLETED SUCCESSFULLY!")
    print("===========================================================================================")

if __name__ == "__main__":
    main()

