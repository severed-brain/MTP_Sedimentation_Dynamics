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
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
struct_dir = os.path.join(mb_custom, "structures")
data_dir = os.path.join(mb_custom, "data")
plots_dir = os.path.join(mb_custom, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(struct_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

dt = 0.02
n_steps = 50
d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]

additional_shapes = {
    "cube": {
        "vertex": "cube_N_56.vertex",
        "N": 56,
        "blob_radius": 0.3333,
        "F": 1000.0,
        "R": 1.0
    },
    "hexagonal_prism": {
        "vertex": "hexagonal_prism_N_74.vertex",
        "N": 74,
        "blob_radius": 0.1601,
        "F": 1000.0,
        "R": 1.0
    },
    "trefoil_knot": {
        "vertex": "trefoil_knot_N_64.vertex",
        "N": 64,
        "blob_radius": 0.1288,
        "F": 1000.0,
        "R": 1.0
    }
}

def run_pair_sim(shape_name, d_r, is_single=False):
    cfg = additional_shapes[shape_name]
    N = cfg["N"]
    brad = cfg["blob_radius"]
    F = cfg["F"]
    g_val = F / N
    R = cfg["R"]
    
    shape_data_dir = os.path.join(data_dir, shape_name)
    sub_tag = "single" if is_single else f"d_{int(d_r)}"
    run_dir = os.path.join(shape_data_dir, sub_tag)
    os.makedirs(run_dir, exist_ok=True)
    
    out_base = f"{shape_name}_{sub_tag}"
    vertex_file = os.path.join(struct_dir, cfg["vertex"])
    clones_file = os.path.join(run_dir, f"{out_base}.clones")
    inp_file = os.path.join(run_dir, f"input_{out_base}.dat")
    config_file = os.path.join(run_dir, f"{out_base}.{out_base}.config")
    
    # Write clones
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
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", os.path.basename(inp_file)]
    t0 = time.time()
    subprocess.run(cmd, cwd=run_dir, env=env, check=True, capture_output=True, text=True)
    wallclock = time.time() - t0
    
    # Parse config
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
    mean_vz = (vz_b1 + vz_b2) / 2.0 if not is_single else vz_b1
    
    return {
        "shape": shape_name,
        "d_r": d_r if not is_single else 0.0,
        "is_single": is_single,
        "vz_b1": vz_b1,
        "vz_b2": vz_b2,
        "mean_vz": mean_vz,
        "wallclock_s": wallclock,
        "times": times,
        "traj_b1": traj_b1,
        "traj_b2": traj_b2
    }

def main():
    print("=" * 95)
    print("PAIR SEDIMENTATION BENCHMARKS FOR CUBE, HEXAGONAL PRISM, AND TREFOIL KNOT")
    print(f"Shapes: Cube (N=56), Hexagonal Prism (N=74), Trefoil Knot (N=64)")
    print(f"Time Step: dt={dt}, Steps: {n_steps}, Separations d/R in: {d_ratios}")
    print("=" * 95)
    
    all_shape_results = {}
    
    for s_name, s_info in additional_shapes.items():
        print(f"\n{'='*40} {s_name.upper()} (N={s_info['N']}) {'='*40}")
        # Run baseline
        single_res = run_pair_sim(s_name, d_r=0.0, is_single=True)
        v0 = single_res["mean_vz"]
        print(f"  Single Body Baseline: V_0 = {v0:.4f} units/s (Wallclock: {single_res['wallclock_s']:.2f} s)")
        
        records = []
        print(f"  {'d/R':<8} {'Vz_Mean (u/s)':<16} {'V/V0 (Ratio)':<14} {'Stokeslet':<12} {'Diff (%)':<10} {'Time (s)':<10}")
        print(f"  {'-'*70}")
        
        pair_dict = {}
        for d_r in d_ratios:
            res = run_pair_sim(s_name, d_r=d_r, is_single=False)
            vz = res["mean_vz"]
            ratio = vz / v0
            theo_stokeslet = 1.0 + 0.75 * (1.0 / d_r)
            diff_pct = abs(ratio - theo_stokeslet) / theo_stokeslet * 100.0
            
            records.append({
                "d_r": d_r,
                "vz_b1": res["vz_b1"],
                "vz_b2": res["vz_b2"],
                "mean_vz": vz,
                "v0": v0,
                "ratio": ratio,
                "theo_stokeslet": theo_stokeslet,
                "diff_pct": diff_pct,
                "wallclock_s": res["wallclock_s"]
            })
            pair_dict[str(d_r)] = res
            print(f"  {d_r:<8.1f} {vz:<16.4f} {ratio:<14.4f} {theo_stokeslet:<12.4f} {diff_pct:<10.2f}% {res['wallclock_s']:<10.2f}")
            
        all_shape_results[s_name] = {
            "v0": v0,
            "single": single_res,
            "pairs": pair_dict,
            "records": records
        }
        
        # Write .dat file for this shape
        dat_path = os.path.join(data_dir, s_name, f"{s_name}_pair_scaling.dat")
        with open(dat_path, "w") as f:
            f.write(f"# RigidMultiblobsWall Pair Sedimentation Scaling: {s_name.replace('_', ' ').title()}\n")
            f.write(f"# Discretization: N={s_info['N']}, Blob Radius a={s_info['blob_radius']}, Force: F={s_info['F']}, dt={dt}\n")
            f.write(f"# Baseline Solitary Velocity: V0 = {v0:.6f} units/s\n")
            f.write("# ----------------------------------------------------------------------------------------------------------------------\n")
            f.write("# d/R      Vz_Body1       Vz_Body2       Vz_Mean        Vz/V0 (Ratio)  Stokeslet_Th   Diff(%)    Wallclock(s)\n")
            f.write("# ----------------------------------------------------------------------------------------------------------------------\n")
            for r in records:
                f.write(f"  {r['d_r']:<7.2f}  {r['vz_b1']:<13.6f}  {r['vz_b2']:<13.6f}  {r['mean_vz']:<13.6f}  {r['ratio']:<13.6f}  {r['theo_stokeslet']:<13.6f}  {r['diff_pct']:<9.2f}  {r['wallclock_s']:<10.2f}\n")
        print(f"  Saved scientific data table to: {dat_path}")
        
    # -------------------------------------------------------------------------
    # MULTI-SHAPE PAIR COMPARISON PLOT
    # -------------------------------------------------------------------------
    print("\nGenerating comprehensive pair scaling comparison plot...")
    plt.figure(figsize=(10, 7))
    
    shape_colors = {"cube": "#1f77b4", "hexagonal_prism": "#2ca02c", "trefoil_knot": "#d62728"}
    shape_markers = {"cube": "s", "hexagonal_prism": "h", "trefoil_knot": "o"}
    shape_labels = {"cube": "Cube (N=56)", "hexagonal_prism": "Hexagonal Prism (N=74)", "trefoil_knot": "Chiral Trefoil Knot (N=64)"}
    
    d_dense = np.linspace(1.8, 11.0, 200)
    theo_curve = 1.0 + 0.75 * (1.0 / d_dense)
    
    for s_name, s_data in all_shape_results.items():
        ds = [r["d_r"] for r in s_data["records"]]
        ratios = [r["ratio"] for r in s_data["records"]]
        plt.plot(ds, ratios, marker=shape_markers[s_name], color=shape_colors[s_name],
                 linewidth=2.2, markersize=8, label=shape_labels[s_name])
                 
    plt.plot(d_dense, theo_curve, 'k--', linewidth=2.0, label=r'Continuum Stokeslet ($1 + \frac{3}{4}\frac{R}{d}$)')
    plt.axhline(1.0, color='gray', linestyle=':', label='Single Body Baseline (1.0)')
    
    plt.title("Hydrodynamic Drafting Factor $V_z / V_0$ vs $d/R$ Across Geometries", fontsize=13, fontweight='bold')
    plt.xlabel("Separation Ratio $d/R$", fontsize=12)
    plt.ylabel("Velocity Enhancement $V_z / V_0$", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "additional_shapes_pair_scaling.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated plot: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "additional_shapes_pair_scaling.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'additional_shapes_pair_scaling.png')}")
    print("=" * 95)
    print("ALL ADDITIONAL PAIR SIMULATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()

