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
disc_mb_dir = os.path.join(base_dir, "custom_simulations", "disc", "multi_body")
struct_dir = os.path.join(disc_mb_dir, "structures")
data_dir = os.path.join(disc_mb_dir, "data")
plots_dir = os.path.join(disc_mb_dir, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(struct_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

# Multi-resolution definitions for disc
resolutions = {
    19: {"blob_radius": 0.2500, "vertex": "disc_N_19.vertex"},
    37: {"blob_radius": 0.1667, "vertex": "disc_N_37.vertex"},
    61: {"blob_radius": 0.1250, "vertex": "disc_N_61.vertex"}
}

R = 1.0
F_total = 1000.0
dt = 0.02
n_steps = 50

# Benchmark distance ratios for resolution comparison
d_ratios = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

def run_simulation(N, d_r, is_single=False):
    """
    Execute native RMBW deterministic_forward_euler for a given resolution N and distance d/R.
    All outputs strictly isolated in data/N_<N>/d_<ratio>/ or data/N_<N>/single/.
    """
    cfg = resolutions[N]
    brad = cfg["blob_radius"]
    g_val = F_total / N
    
    n_dir = os.path.join(data_dir, f"N_{N}")
    sub_tag = "single" if is_single else f"d_{str(d_r).replace('.', '_')}"
    run_dir = os.path.join(n_dir, sub_tag)
    os.makedirs(run_dir, exist_ok=True)
    
    out_base = f"disc_N_{N}_{sub_tag}"
    vertex_file = os.path.join(struct_dir, cfg["vertex"])
    clones_file = os.path.join(run_dir, f"{out_base}.clones")
    inp_file = os.path.join(run_dir, f"input_{out_base}.dat")
    config_file = os.path.join(run_dir, f"{out_base}.{out_base}.config")
    
    # 1. Write .clones
    if is_single:
        clones_content = "1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
    else:
        d = d_r * R
        x1 = -d / 2.0
        x2 = +d / 2.0
        clones_content = (
            f"2\n"
            f"{x1:.6f} 0.0 0.0 1.0 0.0 0.0 0.0\n"
            f"{x2:.6f} 0.0 0.0 1.0 0.0 0.0 0.0\n"
        )
    with open(clones_file, "w") as f:
        f.write(clones_content)
        
    # 2. Write input .dat file (native RMBW syntax)
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
blob_radius                              {brad}
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
        
    # 3. Run solver
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", os.path.basename(inp_file)]
    t0 = time.time()
    subprocess.run(cmd, cwd=run_dir, env=env, check=True, capture_output=True, text=True)
    sim_wallclock = time.time() - t0
    
    # 4. Parse .config
    with open(config_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    traj_b1 = []
    traj_b2 = []
    times = []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        traj_b1.append(bodies[0])
        if nb > 1:
            traj_b2.append(bodies[1])
        times.append(step * dt)
        step += 1
        
    t_span = times[-1] - times[0]
    vz_b1 = abs(traj_b1[-1][2] - traj_b1[0][2]) / t_span
    vz_b2 = abs(traj_b2[-1][2] - traj_b2[0][2]) / t_span if not is_single else 0.0
    vz_avg = (vz_b1 + vz_b2) / 2.0 if not is_single else vz_b1
    
    return {
        "N": N,
        "d_r": d_r if not is_single else 0.0,
        "is_single": is_single,
        "vz_b1": vz_b1,
        "vz_b2": vz_b2,
        "vz_avg": vz_avg,
        "wallclock_s": sim_wallclock,
        "times": times,
        "traj_b1": traj_b1,
        "traj_b2": traj_b2
    }

def main():
    print("=" * 95)
    print("MULTI-RESOLUTION MULTI-BODY DISC SEDIMENTATION SUITE (N = 19, 37, 61)")
    print("RigidMultiblobsWall Forward Euler Integration across Separations d/R")
    print(f"Force: F={F_total}, Time-Step: dt={dt}, Steps: {n_steps}, Distances: {d_ratios}")
    print("=" * 95)
    
    all_records = []
    raw_trajectories_data = {}
    
    for N in [19, 37, 61]:
        print(f"\n{'='*40} RESOLUTION N = {N} {'='*40}")
        # Baseline single disc
        single_res = run_simulation(N, d_r=0.0, is_single=True)
        v0 = single_res["vz_avg"]
        print(f"  -> Single Body Baseline (N={N}): V_0 = {v0:.4f} units/s (Time: {single_res['wallclock_s']:.2f} s)")
        
        raw_trajectories_data[str(N)] = {
            "v0": v0,
            "single": single_res,
            "pairs": {}
        }
        
        print(f"  {'d/R':<8} {'V_z (Mean)':<14} {'V_ratio (V/V0)':<16} {'Stokeslet':<12} {'Diff (%)':<10} {'Wallclock (s)':<12}")
        print(f"  {'-'*75}")
        
        for d_r in d_ratios:
            res = run_simulation(N, d_r=d_r, is_single=False)
            vz = res["vz_avg"]
            ratio = vz / v0
            theory_stokeslet = 1.0 + (2.0 / math.pi) * (1.0 / d_r)
            diff_pct = abs(ratio - theory_stokeslet) / theory_stokeslet * 100.0
            
            all_records.append({
                "N": N,
                "d_r": d_r,
                "Vz_b1": res["vz_b1"],
                "Vz_b2": res["vz_b2"],
                "Vz_mean": vz,
                "V0_baseline": v0,
                "V_ratio": ratio,
                "Stokeslet_theory": theory_stokeslet,
                "Diff_Stokeslet_pct": diff_pct,
                "Wallclock_s": res["wallclock_s"]
            })
            
            raw_trajectories_data[str(N)]["pairs"][str(d_r)] = {
                "vz_avg": vz,
                "V_ratio": ratio,
                "times": res["times"],
                "traj_b1": res["traj_b1"],
                "traj_b2": res["traj_b2"]
            }
            
            print(f"  {d_r:<8.1f} {vz:<14.4f} {ratio:<16.4f} {theory_stokeslet:<12.4f} {diff_pct:<10.2f}% {res['wallclock_s']:<12.2f}")
            
    # Save master scientific .dat file
    dat_path = os.path.join(data_dir, "disc_pair_multires_scaling.dat")
    with open(dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Resolution Multi-Body Sedimentation Scaling: Disc Pairs\n")
        f.write(f"# Discretizations: N in {{19, 37, 61}}, Force: F={F_total}, dt={dt}, eta=1.0\n")
        f.write("# Columns:\n")
        f.write("# 1: N (Number of Blobs)\n")
        f.write("# 2: d/R (Separation Ratio)\n")
        f.write("# 3: Vz_Mean (Mean Terminal Velocity, units/s)\n")
        f.write("# 4: V0_Baseline (Single-Body Baseline Velocity, units/s)\n")
        f.write("# 5: Vz/V0 (Velocity Enhancement Ratio)\n")
        f.write("# 6: Stokeslet_Theory (1 + (2/pi)*(R/d))\n")
        f.write("# 7: Diff_Stokeslet_Pct (%)\n")
        f.write("# 8: Wallclock_Time (s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------------------------\n")
        f.write("# N       d/R      Vz_Mean (u/s)    V0_Base (u/s)    Vz/V0 (Ratio)    Stokeslet_Th     Diff_Stokes(%)   Wallclock(s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------------------------\n")
        for r in all_records:
            f.write(f"  {r['N']:<6d}  {r['d_r']:<7.2f}  {r['Vz_mean']:<15.6f}  {r['V0_baseline']:<15.6f}  {r['V_ratio']:<15.6f}  {r['Stokeslet_theory']:<15.6f}  {r['Diff_Stokeslet_pct']:<15.2f}  {r['Wallclock_s']:<10.2f}\n")
    print(f"\nMaster multi-resolution .dat table saved to: {dat_path}")
    
    # Save trajectories JSON
    json_path = os.path.join(data_dir, "disc_pair_multires_trajectories.json")
    with open(json_path, "w") as f:
        json.dump(raw_trajectories_data, f)
    print(f"Full trajectory JSON saved to: {json_path}")
    
    # -------------------------------------------------------------------------
    # GENERATE 4-PANEL MULTI-RESOLUTION PUBLICATION PLOT
    # -------------------------------------------------------------------------
    print("\nGenerating publication multi-resolution comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    n_colors = {19: "#1f77b4", 37: "#2ca02c", 61: "#d62728"}
    n_markers = {19: "o", 37: "s", 61: "^"}
    
    d_dense = np.linspace(1.8, 11.0, 200)
    theory_disc_stokeslet = 1.0 + (2.0 / math.pi) * (1.0 / d_dense)
    
    # Panel A: Absolute Velocity V_z vs d/R
    ax_a = axes[0, 0]
    for N in [19, 37, 61]:
        sub = [r for r in all_records if r["N"] == N]
        d_vals = [r["d_r"] for r in sub]
        vz_vals = [r["Vz_mean"] for r in sub]
        v0 = sub[0]["V0_baseline"]
        ax_a.plot(d_vals, vz_vals, marker=n_markers[N], color=n_colors[N], linewidth=2.2,
                  markersize=7, label=f"N = {N} (V_0 = {v0:.1f})")
        ax_a.axhline(v0, color=n_colors[N], linestyle=":", alpha=0.6)
    ax_a.set_title("(A) Absolute Terminal Velocity $V_z$ vs Separation $d/R$", fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_a.set_ylabel("Sedimentation Velocity $V_z$ (units/s)", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=9)
    
    # Panel B: Normalized Enhancement V_z / V_0 vs d/R
    ax_b = axes[0, 1]
    for N in [19, 37, 61]:
        sub = [r for r in all_records if r["N"] == N]
        d_vals = [r["d_r"] for r in sub]
        r_vals = [r["V_ratio"] for r in sub]
        ax_b.plot(d_vals, r_vals, marker=n_markers[N], color=n_colors[N], linewidth=2.2,
                  markersize=7, label=f"RMBW Discretization N = {N}")
    ax_b.plot(d_dense, theory_disc_stokeslet, 'k--', linewidth=1.8, label=r'Broadside Stokeslet ($1 + \frac{2}{\pi}\frac{R}{d}$)')
    ax_b.axhline(1.0, color='gray', linestyle=':', label='Solitary Limit (1.0)')
    ax_b.set_title(r"(B) Normalized Enhancement Ratio $V_z / V_0$ vs $d/R$", fontsize=12, fontweight='bold')
    ax_b.set_xlabel("Separation Ratio $d/R$", fontsize=11)
    ax_b.set_ylabel("Enhancement Factor $V_z / V_0$", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=9)
    
    # Panel C: Spatial Trajectory Profiles at d/R = 3.0
    ax_c = axes[1, 0]
    for N in [19, 37, 61]:
        t_data = raw_trajectories_data[str(N)]["pairs"]["3.0"]
        tb1 = np.array(t_data["traj_b1"])
        tb2 = np.array(t_data["traj_b2"])
        ax_c.plot(tb1[:, 0], tb1[:, 2], '-', color=n_colors[N], linewidth=2.0, label=f"N = {N} (Body 1)")
        ax_c.plot(tb2[:, 0], tb2[:, 2], '--', color=n_colors[N], linewidth=2.0)
    ax_c.set_title("(C) Spatial Trajectory Profiles in $x$--$z$ Plane at $d/R = 3.0$", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Lateral Coordinate $x$", fontsize=11)
    ax_c.set_ylabel("Vertical Coordinate $z$", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=9)
    
    # Panel D: Relative Deviation from Stokeslet Theory vs N
    ax_d = axes[1, 1]
    for d_target in [2.0, 4.0, 10.0]:
        sub_d = [r for r in all_records if r["d_r"] == d_target]
        n_list = [r["N"] for r in sub_d]
        diff_list = [r["Diff_Stokeslet_pct"] for r in sub_d]
        ax_d.plot(n_list, diff_list, 'o-', linewidth=2.0, markersize=8, label=f"d/R = {d_target:.1f}")
    ax_d.set_title("(D) Discretization Convergence of Drafting Deviation (%)", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Resolution $N$ (Blobs per Disc)", fontsize=11)
    ax_d.set_ylabel("Relative Deviation from Stokeslet (%)", fontsize=11)
    ax_d.set_xticks([19, 37, 61])
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "disc_pair_multires_convergence.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {fig_path}")
    
    shutil.copyfile(fig_path, os.path.join(thesis_fig_dir, "disc_pair_multires_convergence.png"))
    print(f"Updated thesis figure: {os.path.join(thesis_fig_dir, 'disc_pair_multires_convergence.png')}")
    print("=" * 95)
    print("DISC MULTI-RESOLUTION SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()

