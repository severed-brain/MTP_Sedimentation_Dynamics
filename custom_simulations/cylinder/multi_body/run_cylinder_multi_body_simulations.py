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
data_dir = os.path.join(cyl_mb_dir, "data")
plots_dir = os.path.join(cyl_mb_dir, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(struct_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

# Configuration parameters for primary cylinder simulation
N = 86
R = 1.0  # characteristic semi-length L/2
blob_radius = 0.0742
F_total = 1000.0
g_val = F_total / N
dt = 0.02
n_steps = 50

# Distinct d/R ratios to test
d_ratios = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]

def run_simulation(d_r, is_single=False):
    """
    Run native RMBW deterministic_forward_euler simulation for cylinder pair or single cylinder.
    Isolates all files inside a clean parameter subfolder (d_<ratio>/ or single/).
    """
    tag = "single" if is_single else f"d_{str(d_r).replace('.', '_')}"
    out_base = f"cylinder_pair_{tag}" if not is_single else "cylinder_single"
    
    run_dir = os.path.join(data_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    
    vertex_file = os.path.join(struct_dir, f"cylinder_N_{N}.vertex")
    clones_file = os.path.join(run_dir, f"{out_base}.clones")
    inp_file = os.path.join(run_dir, f"input_{out_base}.dat")
    config_file = os.path.join(run_dir, f"{out_base}.{out_base}.config")
    
    # 1. Write .clones file
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
        
    # 2. Write input .dat file (strictly standard RMBW keywords)
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
        
    # 3. Execute native RMBW solver inside the parameter subfolder
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", os.path.basename(inp_file)]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=run_dir, env=env, check=True, capture_output=True, text=True)
    sim_wallclock = time.time() - t0
    
    # 4. Parse output .config trajectory
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
        
    # Calculate velocities
    t_span = times[-1] - times[0]
    vz_b1 = abs(traj_b1[-1][2] - traj_b1[0][2]) / t_span
    vx_b1 = abs(traj_b1[-1][0] - traj_b1[0][0]) / t_span
    
    vz_b2 = abs(traj_b2[-1][2] - traj_b2[0][2]) / t_span if not is_single else 0.0
    vx_b2 = abs(traj_b2[-1][0] - traj_b2[0][0]) / t_span if not is_single else 0.0
    vz_avg = (vz_b1 + vz_b2) / 2.0 if not is_single else vz_b1
    
    return {
        "d_r": d_r if not is_single else 0.0,
        "is_single": is_single,
        "vz_b1": vz_b1,
        "vx_b1": vx_b1,
        "vz_b2": vz_b2,
        "vx_b2": vx_b2,
        "vz_avg": vz_avg,
        "wallclock_s": sim_wallclock,
        "times": times,
        "traj_b1": traj_b1,
        "traj_b2": traj_b2
    }

def main():
    print("=" * 85)
    print("MULTI-BODY CYLINDER SEDIMENTATION DYNAMICS SUITE")
    print("RigidMultiblobsWall Deterministic Forward Euler Integration")
    print(f"Discretization: N={N}, Blob Radius a={blob_radius}, F_total={F_total}")
    print(f"Time Step: dt={dt}, Total Steps: {n_steps}, Total Duration: T={dt*n_steps:.2f} s")
    print(f"Evaluating Separation Ratios d/R in: {d_ratios}")
    print("=" * 85)
    
    # Run Baseline: Single Cylinder
    print("\n[Step 1] Simulating isolated baseline cylinder (d/R -> infinity)...")
    single_res = run_simulation(d_r=0.0, is_single=True)
    v0 = single_res["vz_avg"]
    print(f"  -> Single Cylinder Terminal Velocity V_0 = {v0:.6f} units/s (Wallclock: {single_res['wallclock_s']:.2f} s)")
    
    # Run Pairs for each d/R
    print("\n[Step 2] Simulating cylinder pairs across d/R ratios...")
    results = [single_res]
    summary_records = []
    
    print("-" * 85)
    print(f"{'d/R':<8} {'V_z (B1)':<12} {'V_z (B2)':<12} {'V_avg':<12} {'V/V_0':<10} {'Stokeslet':<12} {'Diff (%)':<10} {'Time (s)':<8}")
    print("-" * 85)
    
    for d_r in d_ratios:
        res = run_simulation(d_r=d_r, is_single=False)
        results.append(res)
        
        vz_avg = res["vz_avg"]
        ratio = vz_avg / v0
        theory_stokeslet = 1.0 + 0.75 * (1.0 / d_r)
        diff_pct = abs(ratio - theory_stokeslet) / theory_stokeslet * 100.0
        
        summary_records.append({
            "d_r": d_r,
            "Vz_b1": res["vz_b1"],
            "Vz_b2": res["vz_b2"],
            "Vz_avg": vz_avg,
            "Vx_b1": res["vx_b1"],
            "Vx_b2": res["vx_b2"],
            "V_ratio": ratio,
            "Stokeslet_theory": theory_stokeslet,
            "Diff_Stokeslet_pct": diff_pct,
            "Sim_Wallclock_s": res["wallclock_s"]
        })
        
        print(f"{d_r:<8.1f} {res['vz_b1']:<12.4f} {res['vz_b2']:<12.4f} {vz_avg:<12.4f} {ratio:<10.4f} {theory_stokeslet:<12.4f} {diff_pct:<10.2f}% {res['wallclock_s']:<8.2f}")
        
    dat_path = os.path.join(data_dir, "cylinder_pair_scaling.dat")
    with open(dat_path, "w") as f:
        f.write("# RigidMultiblobsWall Multi-Body Sedimentation Scaling: Cylindrical Particle Pairs (N=86)\n")
        f.write(f"# Scheme: deterministic_forward_euler, dt={dt}, eta=1.0, F={F_total}, a={blob_radius}\n")
        f.write(f"# Single Cylinder Baseline: V0 = {v0:.6f} units/s\n")
        f.write("# Columns:\n")
        f.write("# 1: d/R (Separation Ratio)\n")
        f.write("# 2: Vz_Body1 (units/s)\n")
        f.write("# 3: Vz_Body2 (units/s)\n")
        f.write("# 4: Vz_Mean (units/s)\n")
        f.write("# 5: Vz/V0 (Enhancement Ratio)\n")
        f.write("# 6: Stokeslet_Theory (1 + 0.75*(R/d))\n")
        f.write("# 7: Diff_Stokeslet_Pct (%)\n")
        f.write("# 8: Wallclock_Time (s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------------\n")
        f.write("# d/R      Vz_Body1       Vz_Body2       Vz_Mean        Vz/V0      Stokeslet   Diff(%)    Time(s)\n")
        f.write("# ----------------------------------------------------------------------------------------------------------------------\n")
        for rec in summary_records:
            f.write(f"  {rec['d_r']:<7.2f}  {rec['Vz_b1']:<13.6f}  {rec['Vz_b2']:<13.6f}  {rec['Vz_avg']:<13.6f}  {rec['V_ratio']:<9.6f}  {rec['Stokeslet_theory']:<10.6f}  {rec['Diff_Stokeslet_pct']:<9.2f}  {rec['Sim_Wallclock_s']:<7.2f}\n")
    print("-" * 85)
    print(f"Master scaling table saved to: {dat_path}")
    
    # Also save structured JSON for trajectory plotting
    json_path = os.path.join(data_dir, "cylinder_pair_trajectories.json")
    with open(json_path, "w") as f:
        json_dump_data = {
            "v0": float(v0),
            "simulations": [
                {
                    "d_r": float(r["d_r"]),
                    "is_single": bool(r["is_single"]),
                    "vz_avg": float(r["vz_avg"]),
                    "times": r["times"],
                    "traj_b1": r["traj_b1"],
                    "traj_b2": r["traj_b2"]
                }
                for r in results
            ]
        }
        json.dump(json_dump_data, f)
    print(f"Complete trajectory trajectories JSON saved to: {json_path}")
    
    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE 4-PANEL FIGURE
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating publication-grade 4-panel trajectory & scaling figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    color_palette = plt.cm.magma(np.linspace(0.15, 0.85, len(d_ratios)))
    d_color_map = {d: color_palette[i] for i, d in enumerate(d_ratios)}
    
    # Panel A: Spatial Trajectories (x vs z plane)
    ax_a = axes[0, 0]
    for r in results:
        if r["is_single"]:
            continue
        d = r["d_r"]
        tb1 = np.array(r["traj_b1"])
        tb2 = np.array(r["traj_b2"])
        ax_a.plot(tb1[:, 0], tb1[:, 2], '-', color=d_color_map[d], linewidth=2.0, label=f"d/R = {d:.1f}")
        ax_a.plot(tb2[:, 0], tb2[:, 2], '--', color=d_color_map[d], linewidth=2.0)
    ax_a.set_title("(A) Spatial Sedimentation Trajectories ($x$ vs $z$ Plane)", fontsize=12, fontweight='bold')
    ax_a.set_xlabel("Lateral Position $x$", fontsize=11)
    ax_a.set_ylabel("Vertical Position $z$", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8, loc='lower center', ncol=3)
    
    # Panel B: Vertical Position z(t) vs Time
    ax_b = axes[0, 1]
    for r in results:
        if r["is_single"]:
            tb0 = np.array(r["traj_b1"])
            ax_b.plot(r["times"], tb0[:, 2], 'k:', linewidth=2.8, label=f"Single Cylinder ($V_0={v0:.1f}$)")
        else:
            d = r["d_r"]
            tb1 = np.array(r["traj_b1"])
            ax_b.plot(r["times"], tb1[:, 2], '-', color=d_color_map[d], linewidth=2.0, label=f"d/R = {d:.1f}")
    ax_b.set_title("(B) Vertical Position $z(t)$ vs Time", fontsize=12, fontweight='bold')
    ax_b.set_xlabel("Simulation Time $t$ (s)", fontsize=11)
    ax_b.set_ylabel("Vertical Coordinate $z$", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=8, loc='lower left', ncol=2)
    
    # Panel C: Instantaneous Inter-Body Separation Distance d(t) vs Time
    ax_c = axes[1, 0]
    for r in results:
        if r["is_single"]:
            continue
        d = r["d_r"]
        tb1 = np.array(r["traj_b1"])
        tb2 = np.array(r["traj_b2"])
        dist = np.sqrt(np.sum((tb1[:, :3] - tb2[:, :3])**2, axis=1))
        ax_c.plot(r["times"], dist, '-', color=d_color_map[d], linewidth=2.0, label=f"Init $d/R={d:.1f}$")
    ax_c.set_title("(C) Instantaneous Separation Distance $d(t)/R$ vs Time", fontsize=12, fontweight='bold')
    ax_c.set_xlabel("Simulation Time $t$ (s)", fontsize=11)
    ax_c.set_ylabel("Instantaneous Distance $d(t) / R$", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=8, loc='center right', ncol=2)
    
    # Panel D: Sedimentation Velocity Enhancement vs d/R
    ax_d = axes[1, 1]
    d_sim = np.array([r["d_r"] for r in summary_records])
    ratio_sim = np.array([r["V_ratio"] for r in summary_records])
    
    d_dense = np.linspace(1.8, 13.0, 200)
    theory_stokeslet = 1.0 + 0.75 * (1.0 / d_dense)
    
    ax_d.plot(d_sim, ratio_sim, 'mo-', linewidth=2.4, markersize=8, label=r'RMBW Cylinder Multiblob ($N=86$)')
    ax_d.plot(d_dense, theory_stokeslet, 'k--', linewidth=1.8, label=r'Point-Stokeslet Far-Field ($1 + \frac{3}{4}\frac{R}{d}$)')
    ax_d.axhline(1.0, color='gray', linestyle=':', label='Single Cylinder Baseline (1.0)')
    
    ax_d.set_title(r"(D) Cooperative Velocity Enhancement $V_z / V_0$ vs $d/R$", fontsize=12, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $(d / R)$", fontsize=11)
    ax_d.set_ylabel("Velocity Ratio $V_z / V_0$", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    fig_path = os.path.join(plots_dir, "cylinder_pair_trajectories.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {fig_path}")
    
    shutil.copyfile(fig_path, os.path.join(thesis_fig_dir, "cylinder_pair_trajectories.png"))
    print(f"Updated thesis figure: {os.path.join(thesis_fig_dir, 'cylinder_pair_trajectories.png')}")
    print("=" * 85)
    print("CYLINDER PAIR SIMULATIONS AND PLOTS COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    main()
