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
sphere_mb_dir = os.path.join(base_dir, "custom_simulations", "sphere", "multi_body")
struct_dir = os.path.join(sphere_mb_dir, "structures")
data_dir = os.path.join(sphere_mb_dir, "data", "five_body")
plots_dir = os.path.join(sphere_mb_dir, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(struct_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

N = 42
blob_radius = 0.2735
vertex_name = "sphere_N_42.vertex"
F_total = 1000.0
g_val = F_total / N
R = 1.0
dt = 0.02
n_steps = 50
d_ratios = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

# Baseline solitary sphere velocity
V0_BASELINE = 47.276110

def run_simulation(config_type, d_r, clones_coords):
    """
    Run native RMBW deterministic_forward_euler for 5 spheres in either linear_chain or ring_cluster.
    Quarantined into data/five_body/<config_type>/d_<ratio>/.
    """
    tag = f"d_{str(d_r).replace('.', '_')}"
    run_dir = os.path.join(data_dir, config_type, tag)
    os.makedirs(run_dir, exist_ok=True)
    
    out_base = f"sphere_5b_{config_type}_{tag}"
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
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", os.path.basename(inp_file)]
    t0 = time.time()
    subprocess.run(cmd, cwd=run_dir, env=env, check=True, capture_output=True, text=True)
    sim_wallclock = time.time() - t0
    
    # Parse trajectory .config
    with open(config_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    all_trajs = [[] for _ in range(5)]
    times = []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        for b_idx in range(nb):
            all_trajs[b_idx].append([float(x) for x in lines[idx].split()])
            idx += 1
        times.append(step * dt)
        step += 1
        
    t_span = times[-1] - times[0]
    velocities_z = [abs(all_trajs[b][-1][2] - all_trajs[b][0][2]) / t_span for b in range(5)]
    velocities_x = [abs(all_trajs[b][-1][0] - all_trajs[b][0][0]) / t_span for b in range(5)]
    mean_vz = float(np.mean(velocities_z))
    
    return {
        "config_type": config_type,
        "d_r": d_r,
        "velocities_z": velocities_z,
        "velocities_x": velocities_x,
        "mean_vz": mean_vz,
        "enhancement_ratio": mean_vz / V0_BASELINE,
        "wallclock_s": sim_wallclock,
        "times": times,
        "trajectories": all_trajs
    }

def main():
    print("=" * 95)
    print("5-SPHERE MULTI-BODY SEDIMENTATION SUITE (M = 5, N = 42 per sphere)")
    print("RigidMultiblobsWall Forward Euler Integration across Separations d/R")
    print(f"Configurations: Linear Chain (Array) & Pentagonal Ring (Cluster)")
    print(f"Force: F={F_total}, Time-Step: dt={dt}, Steps: {n_steps}, Baseline V0={V0_BASELINE:.4f}")
    print("=" * 95)
    
    # -------------------------------------------------------------------------
    # SUITE 1: 5-SPHERE LINEAR HORIZONTAL CHAIN
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running 5-Sphere Linear Horizontal Chain Simulations...")
    chain_records = []
    chain_trajs = {}
    
    print("-" * 95)
    print(f"{'d/R':<7} {'Vz_Edge (1,5)':<15} {'Vz_Mid (2,4)':<15} {'Vz_Center (3)':<15} {'Vz_Mean':<12} {'Ratio V/V0':<12} {'Wallclock':<10}")
    print("-" * 95)
    
    for d_r in d_ratios:
        d = d_r * R
        # 5 spheres along x-axis centered at x=0: x in {-2d, -d, 0, d, 2d}
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
        
        print(f"{d_r:<7.1f} {v_edge:<15.4f} {v_mid:<15.4f} {v_ctr:<15.4f} {mean_v:<12.4f} {ratio:<12.4f} {res['wallclock_s']:<10.2f} s")
        
    # Write master .dat for linear chain
    chain_dat_path = os.path.join(data_dir, "sphere_5body_chain_scaling.dat")
    with open(chain_dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Body Sedimentation Scaling: 5-Sphere Linear Chain (Array)\n")
        f.write(f"# Discretization: N=42 blobs per sphere, M=5 bodies, Total Blobs = 210, Force: F={F_total}, dt={dt}\n")
        f.write(f"# Single Sphere Baseline: V0 = {V0_BASELINE:.6f} units/s\n")
        f.write("# Spatial Configuration: Symmetrical horizontal chain along x-axis: x_k in {-2d, -d, 0, d, 2d}\n")
        f.write("# --------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("# d/R      Vz_B1(Edge)    Vz_B2(Mid)     Vz_B3(Center)  Vz_B4(Mid)     Vz_B5(Edge)    Vz_Mean        Vz/V0(Ratio)   Center/Edge_Ratio  Wallclock(s)\n")
        f.write("# --------------------------------------------------------------------------------------------------------------------------------------------------------\n")
        for r in chain_records:
            ctr_edge_ratio = r["vz_center"] / r["vz_edge"]
            f.write(f"  {r['d_r']:<7.2f}  {r['vz_b1']:<13.6f}  {r['vz_b2']:<13.6f}  {r['vz_b3']:<13.6f}  {r['vz_b4']:<13.6f}  {r['vz_b5']:<13.6f}  {r['vz_mean']:<13.6f}  {r['enhancement_ratio']:<13.6f}  {ctr_edge_ratio:<17.6f}  {r['wallclock_s']:<10.2f}\n")
    print(f"\nSaved linear chain scientific data table to: {chain_dat_path}")
    
    # -------------------------------------------------------------------------
    # SUITE 2: 5-SPHERE PENTAGONAL RING CLUSTER
    # -------------------------------------------------------------------------
    print("\n[PART 2] Running 5-Sphere Pentagonal Ring Cluster Simulations...")
    ring_records = []
    ring_trajs = {}
    
    print("-" * 95)
    print(f"{'d/R':<7} {'Vz_Individual (Mean of 5)':<28} {'Vz_Mean':<14} {'Cluster Ratio V/V0':<22} {'Wallclock':<10}")
    print("-" * 95)
    
    # Chord length is d. Radius of circumcircle: R_ring = d / (2 * sin(pi / 5))
    sin_pi_5 = math.sin(math.pi / 5.0) # sin(36 deg) ~ 0.587785
    
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
    ring_dat_path = os.path.join(data_dir, "sphere_5body_ring_scaling.dat")
    with open(ring_dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Body Sedimentation Scaling: 5-Sphere Pentagonal Ring (Cluster)\n")
        f.write(f"# Discretization: N=42 blobs per sphere, M=5 bodies, Total Blobs = 210, Force: F={F_total}, dt={dt}\n")
        f.write(f"# Single Sphere Baseline: V0 = {V0_BASELINE:.6f} units/s\n")
        f.write("# Spatial Configuration: Symmetrical pentagonal ring on xy-plane with inter-body chord distance d\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------\n")
        f.write("# d/R      R_ring/R   Vz_Cluster (u/s)   V0_Baseline (u/s)  Cluster_Ratio V/V0 Theoretical_Stokeslet(Est) Wallclock(s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------\n")
        for r in ring_records:
            # 5-body Stokeslet cluster approximation: 1 + (M-1) * (3/4) * (R/d_eff)
            # pairwise chords: 2 at d, 2 at 2*d*sin(72)/sin(36) = 1.618*d
            # Stokeslet sum = 2*(3/4)*(R/d) + 2*(3/4)*(R/(1.618*d)) = (3/2)*(1 + 1/1.618)*(R/d) = 2.427 * (R/d)
            theo_cluster = 1.0 + 2.427 * (1.0 / r["d_r"])
            f.write(f"  {r['d_r']:<7.2f}  {r['r_ring']:<9.4f}  {r['vz_mean']:<17.6f}  {V0_BASELINE:<17.6f}  {r['enhancement_ratio']:<19.6f}  {theo_cluster:<27.6f}  {r['wallclock_s']:<10.2f}\n")
    print(f"\nSaved ring cluster scientific data table to: {ring_dat_path}")
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE 4-PANEL FIGURE FOR 5-BODY DYNAMICS
    # -------------------------------------------------------------------------
    print("\nGenerating publication 4-panel figure for 5-sphere sedimentation...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(d_ratios)))
    d_color = {d: colors[i] for i, d in enumerate(d_ratios)}
    
    # Panel A: Linear Chain Spatial Profile (Instantaneous sag / deformation)
    ax_a = axes[0, 0]
    for d_r in [2.0, 3.0, 4.0, 8.0]:
        sim = chain_trajs[str(d_r)]
        t_final_idx = -1
        xs = [sim["trajectories"][b][t_final_idx][0] for b in range(5)]
        zs = [sim["trajectories"][b][t_final_idx][2] for b in range(5)]
        ax_a.plot(xs, zs, 'o-', color=d_color[d_r], linewidth=2.2, markersize=8, label=f"d/R = {d_r:.1f}")
    ax_a.set_title("(A) 5-Sphere Chain Deformation Profile (at $t = 1.0$ s)", fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Lateral Position $x$", fontsize=11)
    ax_a.set_ylabel("Vertical Position $z$", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=9)
    
    # Panel B: Velocity Gradient Across Bodies in Chain (Center vs Edge)
    ax_b = axes[0, 1]
    body_indices = [1, 2, 3, 4, 5]
    for d_r in [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
        sim = chain_trajs[str(d_r)]
        vzs = sim["velocities_z"]
        ax_b.plot(body_indices, vzs, 's-', color=d_color[d_r], linewidth=2.0, markersize=7, label=f"d/R = {d_r:.1f}")
    ax_b.axhline(V0_BASELINE, color='gray', linestyle=':', label='Single Sphere Baseline')
    ax_b.set_title("(B) Velocity Gradient Across Chain Bodies (Center Speedup)", fontsize=12, fontweight='bold')
    ax_b.set_xlabel("Sphere Index $k$ (1=Left Edge, 3=Center, 5=Right Edge)", fontsize=11)
    ax_b.set_ylabel("Sedimentation Velocity $V_z$ (units/s)", fontsize=11)
    ax_b.set_xticks(body_indices)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=8, ncol=2)
    
    # Panel C: Ring Cluster Collective Speedup vs d/R
    ax_c = axes[1, 0]
    d_vals = [r["d_r"] for r in ring_records]
    ring_ratios = [r["enhancement_ratio"] for r in ring_records]
    chain_ratios = [r["enhancement_ratio"] for r in chain_records]
    
    d_dense = np.linspace(1.8, 11.0, 200)
    pair_theory = 1.0 + 0.75 * (1.0 / d_dense)
    ring_theory = 1.0 + 2.427 * (1.0 / d_dense)
    
    ax_c.plot(d_vals, ring_ratios, 'ro-', linewidth=2.4, markersize=8, label='5-Sphere Ring Cluster (RMBW)')
    ax_c.plot(d_vals, chain_ratios, 'bd--', linewidth=2.2, markersize=7, label='5-Sphere Linear Chain (Mean)')
    ax_c.plot(d_dense, ring_theory, 'k--', linewidth=1.8, label=r'5-Body Stokeslet Screening Theory ($1 + 2.43 \frac{R}{d}$)')
    ax_c.plot(d_dense, pair_theory, 'g:', linewidth=1.6, label=r'2-Body Pair Baseline ($1 + 0.75 \frac{R}{d}$)')
    ax_c.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax_c.set_title(r"(C) Collective Drafting Factor $V_z / V_0$ vs $d/R$ ($M = 5$ vs $M = 2$)", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_c.set_ylabel("Collective Velocity Ratio $V_z / V_0$", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=8, loc='upper right')
    
    # Panel D: Center-to-Edge Velocity Ratio vs d/R
    ax_d = axes[1, 1]
    ctr_edge_ratios = [r["vz_center"] / r["vz_edge"] for r in chain_records]
    ax_d.plot(d_vals, ctr_edge_ratios, 'mo-', linewidth=2.4, markersize=8, label=r'Chain Sagging Factor ($V_{\mathrm{center}} / V_{\mathrm{edge}}$)')
    ax_d.axhline(1.0, color='gray', linestyle=':', label='Rigid/Decoupled Array (1.0)')
    ax_d.set_title(r"(D) Hydrodynamic Screening Gradient ($V_{\mathrm{center}} / V_{\mathrm{edge}}$)", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_d.set_ylabel(r"Center-to-Edge Ratio $V_z(3) / V_z(1)$", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "sphere_5body_dynamics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "sphere_5body_dynamics.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'sphere_5body_dynamics.png')}")
    print("=" * 95)
    print("5-SPHERE MULTI-BODY SIMULATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()

