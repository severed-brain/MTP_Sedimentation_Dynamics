import os
import sys
import math
import time
import json
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
cyl_mb_dir = os.path.join(base_dir, "custom_simulations", "cylinder", "multi_body")
struct_dir = os.path.join(cyl_mb_dir, "structures")
data_dir = os.path.join(cyl_mb_dir, "data", "five_body")
plots_dir = os.path.join(cyl_mb_dir, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(struct_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

N = 86
blob_radius = 0.0742
vertex_name = "cylinder_N_86.vertex"
F_total = 1000.0
g_val = F_total / N
R = 1.0
dt = 0.02
n_steps = 50
d_ratios = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

# Baseline solitary cylinder velocity
V0_BASELINE = 98.126625

def run_simulation(config_type, d_r, clones_coords):
    """
    Run native RMBW deterministic_forward_euler for 5 cylinders in either linear_chain or ring_cluster.
    Quarantined into data/five_body/<config_type>/d_<ratio>/.
    """
    tag = f"d_{str(d_r).replace('.', '_')}"
    run_dir = os.path.join(data_dir, config_type, tag)
    os.makedirs(run_dir, exist_ok=True)
    
    out_base = f"cylinder_5b_{config_type}_{tag}"
    vertex_file = os.path.join(struct_dir, vertex_name)
    clones_file = os.path.join(run_dir, f"{out_base}.clones")
    inp_file = os.path.join(run_dir, f"input_{out_base}.dat")
    config_file = os.path.join(run_dir, f"{out_base}.{out_base}.config")
    
    # Write clones file for 5 bodies
    with open(clones_file, "w") as f:
        f.write(f"{len(clones_coords)}\n")
        for c in clones_coords:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} 1.0 0.0 0.0 0.0\n")
            
    # Write input .dat file
    inp_content = f"""scheme                                   deterministic_forward_euler
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        {g_val:.14f}
blob_radius                              {blob_radius}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_base}
structure {os.path.abspath(vertex_file)} {os.path.abspath(clones_file)}
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    # Execute native RMBW solver
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", os.path.basename(inp_file)]
    t0 = time.time()
    subprocess.run(cmd, cwd=run_dir, env=env, check=True, capture_output=True, text=True)
    sim_wallclock = time.time() - t0
    
    # Parse trajectory
    with open(config_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    trajectories = [[] for _ in range(len(clones_coords))]
    times = []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        for b_idx in range(nb):
            coords = [float(x) for x in lines[idx].split()]
            trajectories[b_idx].append(coords)
            idx += 1
        times.append(step * dt)
        step += 1
        
    # Compute velocity for each body
    t_span = times[-1] - times[0]
    velocities_z = [abs(t[-1][2] - t[0][2]) / t_span for t in trajectories]
    mean_vz = float(np.mean(velocities_z))
    ratio = mean_vz / V0_BASELINE
    
    return {
        "config_type": config_type,
        "d_r": d_r,
        "velocities_z": velocities_z,
        "mean_vz": mean_vz,
        "enhancement_ratio": ratio,
        "times": times,
        "trajectories": trajectories,
        "wallclock_s": sim_wallclock
    }

def main():
    print("=" * 95)
    print("FIVE-BODY CYLINDER MULTI-BODY SEDIMENTATION SUITE (M = 5, N = 86)")
    print("RigidMultiblobsWall Forward Euler Integration across Separations d/R")
    print(f"Discretization: N={N}, Blob Radius: a={blob_radius}, Force/Body: F={F_total}")
    print(f"Time Step: dt={dt}, Steps: {n_steps}, Total Duration: T={dt*n_steps:.2f} s")
    print(f"Separation Ratios d/R: {d_ratios}")
    print("=" * 95)
    
    # -------------------------------------------------------------------------
    # SUITE 1: 5-CYLINDER HORIZONTAL LINEAR CHAIN (ARRAY)
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running 5-Cylinder Linear Horizontal Chain Simulations...")
    chain_records = []
    chain_trajs = {}
    
    print("-" * 95)
    print(f"{'d/R':<7} {'Vz_Edge (B1,B5)':<16} {'Vz_Mid (B2,B4)':<16} {'Vz_Center (B3)':<16} {'Vz_Mean':<12} {'Ratio V/V0':<12} {'Wallclock':<10}")
    print("-" * 95)
    
    for d_r in d_ratios:
        d = d_r * R
        coords = [
            (-2.0 * d, 0.0, 0.0),
            (-1.0 * d, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0 * d, 0.0, 0.0),
            (2.0 * d, 0.0, 0.0)
        ]
        res = run_simulation("linear_chain", d_r, coords)
        v_edge = (res["velocities_z"][0] + res["velocities_z"][4]) / 2.0
        v_mid = (res["velocities_z"][1] + res["velocities_z"][3]) / 2.0
        v_ctr = res["velocities_z"][2]
        mean_v = res["mean_vz"]
        ratio = res["enhancement_ratio"]
        
        chain_records.append({
            "d_r": d_r,
            "vz_b1": res["velocities_z"][0],
            "vz_b2": res["velocities_z"][1],
            "vz_b3": res["velocities_z"][2],
            "vz_b4": res["velocities_z"][3],
            "vz_b5": res["velocities_z"][4],
            "vz_edge": v_edge,
            "vz_mid": v_mid,
            "vz_center": v_ctr,
            "vz_mean": mean_v,
            "enhancement_ratio": ratio,
            "wallclock_s": res["wallclock_s"]
        })
        chain_trajs[str(d_r)] = res
        
        print(f"{d_r:<7.1f} {v_edge:<16.4f} {v_mid:<16.4f} {v_ctr:<16.4f} {mean_v:<12.4f} {ratio:<12.4f} {res['wallclock_s']:<10.2f} s")
        
    # Write master .dat for linear chain
    chain_dat_path = os.path.join(data_dir, "cylinder_5body_chain_scaling.dat")
    with open(chain_dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Body Sedimentation Scaling: 5-Cylinder Linear Chain (Array)\n")
        f.write(f"# Discretization: N=86 blobs per cylinder, M=5 bodies, Total Blobs = 430, Force: F={F_total}, dt={dt}\n")
        f.write(f"# Single Cylinder Baseline: V0 = {V0_BASELINE:.6f} units/s\n")
        f.write("# Spatial Configuration: Symmetrical horizontal chain along x-axis: x_k in {-2d, -d, 0, d, 2d}\n")
        f.write("# Columns:\n")
        f.write("# 1: d/R (Separation Ratio)\n")
        f.write("# 2-6: Vz_B1 to Vz_B5 (units/s)\n")
        f.write("# 7: Vz_Mean (units/s)\n")
        f.write("# 8: Vz/V0 (Enhancement Ratio)\n")
        f.write("# 9: Center/Edge_Ratio (Chain Instability Gradient)\n")
        f.write("# 10: Wallclock_Time (s)\n")
        f.write("# --------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("# d/R      Vz_B1(Edge)    Vz_B2(Mid)     Vz_B3(Center)  Vz_B4(Mid)     Vz_B5(Edge)    Vz_Mean        Vz/V0(Ratio)   Center/Edge_Ratio  Wallclock(s)\n")
        f.write("# --------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        for r in chain_records:
            ctr_edge_ratio = r["vz_center"] / r["vz_edge"]
            f.write(f"  {r['d_r']:<7.2f}  {r['vz_b1']:<13.6f}  {r['vz_b2']:<13.6f}  {r['vz_b3']:<13.6f}  {r['vz_b4']:<13.6f}  {r['vz_b5']:<13.6f}  {r['vz_mean']:<13.6f}  {r['enhancement_ratio']:<13.6f}  {ctr_edge_ratio:<17.6f}  {r['wallclock_s']:<10.2f}\n")
    print(f"\nSaved linear chain scientific data table to: {chain_dat_path}")
    
    # -------------------------------------------------------------------------
    # SUITE 2: 5-CYLINDER PENTAGONAL RING CLUSTER
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running 5-Cylinder Pentagonal Ring Cluster Simulations...")
    ring_records = []
    ring_trajs = {}
    
    print("-" * 95)
    print(f"{'d/R':<7} {'Vz_Individual (Mean of 5)':<28} {'Vz_Mean':<14} {'Cluster Ratio V/V0':<22} {'Wallclock':<10}")
    print("-" * 95)
    
    sin_pi_5 = math.sin(math.pi / 5.0)
    
    for d_r in d_ratios:
        d = d_r * R
        r_ring = d / (2.0 * sin_pi_5)
        coords = []
        for k in range(5):
            phi = 2.0 * math.pi * k / 5.0
            coords.append((r_ring * math.cos(phi), r_ring * math.sin(phi), 0.0))
            
        res = run_simulation("ring_cluster", d_r, coords)
        mean_v = res["mean_vz"]
        ratio = res["enhancement_ratio"]
        
        ring_records.append({
            "d_r": d_r,
            "r_ring": r_ring,
            "vz_mean": mean_v,
            "enhancement_ratio": ratio,
            "wallclock_s": res["wallclock_s"]
        })
        ring_trajs[str(d_r)] = res
        print(f"{d_r:<7.1f} {mean_v:<28.4f} {mean_v:<14.4f} {ratio:<22.4f} {res['wallclock_s']:<10.2f} s")
        
    # Write master .dat for ring cluster
    ring_dat_path = os.path.join(data_dir, "cylinder_5body_ring_scaling.dat")
    with open(ring_dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Body Sedimentation Scaling: 5-Cylinder Pentagonal Ring (Cluster)\n")
        f.write(f"# Discretization: N=86 blobs per cylinder, M=5 bodies, Total Blobs = 430, Force: F={F_total}, dt={dt}\n")
        f.write(f"# Single Cylinder Baseline: V0 = {V0_BASELINE:.6f} units/s\n")
        f.write("# Spatial Configuration: Symmetrical pentagonal ring on xy-plane with inter-body chord distance d\n")
        f.write("# Columns:\n")
        f.write("# 1: d/R (Chord Separation Ratio)\n")
        f.write("# 2: R_ring/R (Circumcircle Radius Ratio)\n")
        f.write("# 3: Vz_Cluster (Mean Sedimentation Velocity, units/s)\n")
        f.write("# 4: V0_Baseline (Single-Body Baseline Velocity, units/s)\n")
        f.write("# 5: Cluster_Ratio (V/V0 Enhancement)\n")
        f.write("# 6: Theoretical_Stokeslet_Est (1 + 2.427*(R/d))\n")
        f.write("# 7: Wallclock_Time (s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------\n")
        f.write("# d/R      R_ring/R   Vz_Cluster (u/s)   V0_Baseline (u/s)  Cluster_Ratio V/V0 Theoretical_Stokeslet(Est) Wallclock(s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------\n")
        for r in ring_records:
            theo_cluster = 1.0 + 2.427 * (1.0 / r["d_r"])
            f.write(f"  {r['d_r']:<7.2f}  {r['r_ring']:<9.4f}  {r['vz_mean']:<17.6f}  {V0_BASELINE:<17.6f}  {r['enhancement_ratio']:<18.6f}  {theo_cluster:<24.6f}  {r['wallclock_s']:<10.2f}\n")
    print(f"\nSaved ring cluster scientific data table to: {ring_dat_path}")
    
    # -------------------------------------------------------------------------
    # GENERATE 4-PANEL COLLECTIVE DYNAMICS PUBLICATION PLOT
    # -------------------------------------------------------------------------
    print("\nGenerating 4-panel publication collective dynamics plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: Horizontal chain deformation profile
    ax_a = axes[0, 0]
    for d_target in [2.0, 3.0, 4.0, 6.0]:
        t_data = chain_trajs[str(d_target)]
        final_coords = [t_data["trajectories"][k][-1] for k in range(5)]
        x_pts = [pt[0] for pt in final_coords]
        z_pts = [pt[2] for pt in final_coords]
        ax_a.plot(x_pts, z_pts, 'o-', linewidth=2.0, markersize=8, label=f"d/R = {d_target:.1f}")
    ax_a.set_title("(A) 5-Cylinder Linear Chain Deformation Profile (Inverted Sagging)", fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Lateral Position $x$", fontsize=11)
    ax_a.set_ylabel("Vertical Position $z$ at $t = 1.0\\text{ s}$", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=9)
    
    # Panel B: Center-to-Edge Velocity Gradient across Chain Bodies
    ax_b = axes[0, 1]
    body_indices = [1, 2, 3, 4, 5]
    for d_target in [2.0, 3.0, 4.0, 6.0, 10.0]:
        t_data = chain_trajs[str(d_target)]
        vels = t_data["velocities_z"]
        ax_b.plot(body_indices, vels, 's-', linewidth=2.0, markersize=7, label=f"d/R = {d_target:.1f}")
    ax_b.set_title("(B) Velocity Distribution Across 5-Cylinder Linear Chain", fontsize=12, fontweight='bold')
    ax_b.set_xlabel("Particle Index in Array ($k = 1 \\dots 5$)", fontsize=11)
    ax_b.set_ylabel("Terminal Velocity $V_z$ (units/s)", fontsize=11)
    ax_b.set_xticks(body_indices)
    ax_b.set_xticklabels(["Edge 1", "Mid 2", "Center 3", "Mid 4", "Edge 5"])
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=9)
    
    # Panel C: Collective Acceleration: Ring Cluster vs Linear Chain vs 2-Body
    ax_c = axes[1, 0]
    d_list = [r["d_r"] for r in ring_records]
    ring_ratios = [r["enhancement_ratio"] for r in ring_records]
    chain_ratios = [r["enhancement_ratio"] for r in chain_records]
    
    # Pair scaling baseline
    cyl_pair_dat = os.path.join(cyl_mb_dir, "data", "cylinder_pair_scaling.dat")
    if os.path.exists(cyl_pair_dat):
        p_data = np.loadtxt(cyl_pair_dat, comments="#")
        ax_c.plot(p_data[:, 0], p_data[:, 4], 'k--', linewidth=1.8, label="2-Body Pair Baseline (M=2)")
        
    ax_c.plot(d_list, ring_ratios, 'ro-', linewidth=2.4, markersize=8, label="5-Cylinder Ring Cluster (M=5)")
    ax_c.plot(d_list, chain_ratios, 'bs-', linewidth=2.4, markersize=8, label="5-Cylinder Linear Chain (M=5)")
    ax_c.axhline(1.0, color='gray', linestyle=':', label='Single Cylinder (1.0)')
    ax_c.set_title("(C) Collective Drafting: 5-Cylinder Ring Cluster vs Linear Chain", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Inter-Body Separation Ratio $d/R$", fontsize=11)
    ax_c.set_ylabel("Velocity Enhancement $V_z / V_0$", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=9)
    
    # Panel D: Center-to-Edge Sagging Instability Ratio vs d/R
    ax_d = axes[1, 1]
    sag_ratios = [r["vz_center"] / r["vz_edge"] for r in chain_records]
    ax_d.plot(d_list, sag_ratios, 'd-', color='#8c564b', linewidth=2.4, markersize=8, label="Sagging Gradient $V_{\\text{center}} / V_{\\text{edge}}$")
    ax_d.axhline(1.0, color='k', linestyle='--', alpha=0.7, label="Uniform Velocity (No Gradient)")
    ax_d.set_title("(D) Chain Instability Gradient ($V_{\\text{center}} / V_{\\text{edge}}$) vs $d/R$", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_d.set_ylabel("Velocity Ratio $V_{\\text{center}} / V_{\\text{edge}}$", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "cylinder_5body_dynamics.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {fig_path}")
    
    shutil.copyfile(fig_path, os.path.join(thesis_fig_dir, "cylinder_5body_dynamics.png"))
    print(f"Updated thesis figure: {os.path.join(thesis_fig_dir, 'cylinder_5body_dynamics.png')}")
    print("=" * 95)
    print("FIVE-BODY CYLINDER SIMULATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()
