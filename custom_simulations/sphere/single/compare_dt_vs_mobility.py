import os
import sys
import math
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
base_dir = r"d:\sedimentation_dynamics"
shell_dir = os.path.join(base_dir, "custom_simulations", "shell")
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")

Ns = [12, 42, 162, 642, 2562]
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

vertex_files = {
    12: "shell_N_12_Rg_1_Rh_1_2625.vertex",
    42: "shell_N_42_Rg_1_Rh_1_1220.vertex",
    162: "shell_N_162_Rg_1_Rh_1_0530.vertex",
    642: "shell_N_642_Rg_1_Rh_1_0239.vertex",
    2562: "shell_N_2562_Rg_1_Rh_1_0113.vertex"
}

def run_simulation_task(task_args):
    scheme, N, F, dt, rerun = task_args
    scheme_str = "deterministic_forward_euler" if scheme == "forward_euler" else "deterministic_adams_bashforth"
    custom_dir = os.path.join(shell_dir, scheme)
    n_dir = os.path.join(custom_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    
    data_dir = os.path.join(n_dir, f"data_dt_{dt}")
    os.makedirs(data_dir, exist_ok=True)
    
    # Ensure vertex and clones exist
    v_src = os.path.join(multi_bodies_dir, "Structures", vertex_files[N])
    v_dst = os.path.join(n_dir, vertex_files[N])
    if not os.path.exists(v_dst):
        import shutil
        shutil.copy(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    if not os.path.exists(c_dst):
        with open(c_dst, "w") as f:
            f.write("1\n0 0 0 1 0 0 0\n")
            
    g = F / N
    out_name = f"data_dt_{dt}/shell_N_{N}_F_{int(F)}"
    inp_name = f"input_F_{int(F)}_dt_{dt}.dat"
    inp_file = os.path.join(n_dir, inp_name)
    cfg_file = os.path.join(n_dir, f"{out_name}.single_body.config")
    
    optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
    steps = 3 if N >= 642 else 5
    
    if rerun or not os.path.exists(cfg_file):
        inp_content = f"""scheme                                   {scheme_str}
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {steps}
n_save                                   1
eta                                      1.0
g                                        {g}
blob_radius                              {optimal_blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_name}
structure {vertex_files[N]} single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"
        
        cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", inp_name]
        subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
        
    # Parse velocity from cfg_file
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    frames = []
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        frames.append(bodies)
        
    z0 = frames[0][0][2]
    z_end = frames[-1][0][2]
    num_frames = len(frames) - 1
    vz = abs((z_end - z0) / (num_frames * dt))
    
    return (scheme, N, F, dt, vz)

def parse_mobility_velocity(N, F):
    vel_file = os.path.join(shell_dir, "mobility", f"N_{N}", "data", f"shell_N_{N}_F_{int(F)}.velocity.dat")
    if not os.path.exists(vel_file):
        # Run mobility if not found
        g = F / N
        out_name = f"data/shell_N_{N}_F_{int(F)}"
        n_dir = os.path.join(shell_dir, "mobility", f"N_{N}")
        os.makedirs(n_dir, exist_ok=True)
        os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
        v_dst = os.path.join(n_dir, vertex_files[N])
        if not os.path.exists(v_dst):
            import shutil
            shutil.copy(os.path.join(multi_bodies_dir, "Structures", vertex_files[N]), v_dst)
        c_dst = os.path.join(n_dir, "single_body.clones")
        if not os.path.exists(c_dst):
            with open(c_dst, "w") as f:
                f.write("1\n0 0 0 1 0 0 0\n")
        optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
        inp_content = f"""scheme                                   mobility
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
eta                                      1.0
g                                        {g}
blob_radius                              {optimal_blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
output_name                              {out_name}
structure {vertex_files[N]} single_body.clones
"""
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        with open(inp_file, "w") as f:
            f.write(inp_content)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"
        cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
        subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)

    vel_data = np.loadtxt(vel_file)
    if vel_data.ndim == 1:
        return abs(vel_data[2])
    return abs(vel_data[0, 2])

def main():
    parser = argparse.ArgumentParser(description="Compare dt=0.01 time-stepping schemes with Mobility scheme.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step dt (default: 0.01)")
    parser.add_argument("--rerun", action="store_true", help="Force rerun simulations")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker count")
    args = parser.parse_args()
    
    dt = args.dt
    print("=" * 80)
    print(f"MULTI-SCHEME HYDRODYNAMIC COMPARISON: dt = {dt} vs MOBILITY")
    print("=" * 80)
    
    # Collect Mobility results
    print("\n[1/3] Loading Mobility Scheme Baseline Results...")
    mobility_results = {N: {} for N in Ns}
    for N in Ns:
        for F in F_values:
            mobility_results[N][F] = parse_mobility_velocity(N, F)
            
    # Prepare simulation tasks for Forward Euler and Adams-Bashforth
    tasks = []
    schemes = ["forward_euler", "adams_bashforth"]
    for scheme in schemes:
        for N in Ns:
            for F in F_values:
                tasks.append((scheme, N, F, dt, args.rerun))
                
    print(f"\n[2/3] Executing {len(tasks)} time-stepping simulations (dt = {dt}) with {args.workers} workers...")
    t0 = time.time()
    results = {s: {N: {} for N in Ns} for s in schemes}
    
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_simulation_task, t) for t in tasks]
        for f in as_completed(futures):
            scheme, N, F, _, vz = f.result()
            results[scheme][N][F] = vz
            print(f"  -> Finished {scheme:16s} N={N:4d} F={int(F):4d} => V_z = {vz:10.5f}")
            
    print(f"All simulations finished in {time.time()-t0:.2f} seconds.\n")
    
    # 3. Print Comprehensive Comparison Table
    print("[3/3] COMPARISON SUMMARY TABLE")
    header = f"{'N':>5} | {'F (Force)':>9} | {'Stokes Law':>11} | {'Mobility':>11} | {'Euler (dt=0.01)':>15} | {'AB (dt=0.01)':>15} | {'|Euler-Mob|':>12} | {'|AB-Mob|':>12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for N in Ns:
        for F in F_values:
            v_stokes = F / (6 * math.pi * 1.0 * 1.0)
            v_mob = mobility_results[N][F]
            v_fe = results["forward_euler"][N][F]
            v_ab = results["adams_bashforth"][N][F]
            diff_fe = abs(v_fe - v_mob)
            diff_ab = abs(v_ab - v_mob)
            print(f"{N:5d} | {int(F):9d} | {v_stokes:11.4f} | {v_mob:11.4f} | {v_fe:15.4f} | {v_ab:15.4f} | {diff_fe:12.2e} | {diff_ab:12.2e}")
        print("-" * len(header))
        
    # Generate Plots
    colors = {12: "#1f77b4", 42: "#ff7f0e", 162: "#2ca02c", 642: "#d62728", 2562: "#9467bd"}
    all_forces = np.linspace(0, max(F_values), 100)
    theoretical_v = all_forces / (6 * np.pi * 1.0 * 1.0)
    
    # -------------------------------------------------------------
    # Plot 1: 3-Panel Side-by-Side Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    all_schemes = ["adams_bashforth", "forward_euler", "mobility"]
    titles = {
        "adams_bashforth": f"Adams-Bashforth (2nd Order, dt={dt})",
        "forward_euler": f"Forward Euler (1st Order, dt={dt})",
        "mobility": "Mobility (Quasi-Steady Direct)"
    }
    
    for ax_idx, s in enumerate(all_schemes):
        ax = axes[ax_idx]
        for N in Ns:
            forces = F_values
            if s == "mobility":
                vels = [mobility_results[N][F] for F in forces]
            else:
                vels = [results[s][N][F] for F in forces]
            ax.plot(forces, vels, marker='o', label=f'N = {N}', color=colors[N], linewidth=1.8, markersize=5)
            
        ax.plot(all_forces, theoretical_v, '--', color='black', label='Stokes Law (Theory)', linewidth=1.5)
        ax.set_title(titles[s], fontsize=12, fontweight='bold')
        ax.set_xlabel('Total Force (N · g)', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc='upper left')
        
    plt.tight_layout()
    p1 = os.path.join(shell_dir, f"schemes_side_by_side_dt_{dt}.png")
    plt.savefig(p1, dpi=300)
    print(f"\nPlot 1 saved: {p1}")
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 2: Direct Overlay and Resolution Convergence vs Stokes
    # -------------------------------------------------------------
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    markers = {"adams_bashforth": "o", "forward_euler": "s", "mobility": "^"}
    linestyles = {"adams_bashforth": "-", "forward_euler": "--", "mobility": ":"}
    
    for N in Ns:
        for s in all_schemes:
            forces = F_values
            vels = [mobility_results[N][F] if s == "mobility" else results[s][N][F] for F in forces]
            label = f'N={N} ({s[:3].upper()})' if N in [12, 2562] else None
            ax1.plot(forces, vels, marker=markers[s], linestyle=linestyles[s],
                     color=colors[N], label=label, alpha=0.85, markersize=5)
                     
    ax1.plot(all_forces, theoretical_v, '--', color='black', linewidth=2.0, label='Stokes Law')
    ax1.set_title(f'Direct Overlay: AB vs Euler (dt={dt}) vs Mobility', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Total Force (N · g)', fontsize=11)
    ax1.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)
    
    # Panel B: Resolution Convergence at F=1000
    ref_F = 1000.0
    stokes_ref = ref_F / (6 * math.pi)
    for s in all_schemes:
        vels_at_ref = [mobility_results[N][ref_F] if s == "mobility" else results[s][N][ref_F] for N in Ns]
        ax2.plot(Ns, vels_at_ref, marker=markers[s], linestyle='-',
                 label=titles[s].split("(")[0].strip(), linewidth=1.8, markersize=7)
                 
    ax2.axhline(stokes_ref, color='black', linestyle='--', linewidth=1.5, label=f'Stokes Law ({stokes_ref:.2f})')
    ax2.set_xscale('log')
    ax2.set_title(f'Velocity Convergence vs Resolution N (Force = {ref_F:.0f})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Discretization Resolution (N blobs)', fontsize=11)
    ax2.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
    ax2.set_xticks(Ns)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    p2 = os.path.join(shell_dir, f"schemes_comparison_dt_{dt}.png")
    plt.savefig(p2, dpi=300)
    print(f"Plot 2 saved: {p2}")
    plt.close()
    
    # -------------------------------------------------------------
    # Plot 3: Time-step sensitivity (dt=0.01 vs dt=0.05 vs Mobility) & Error Residuals
    # -------------------------------------------------------------
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Try to load dt=0.05 results if available
    dt05_results = {"forward_euler": {}, "adams_bashforth": {}}
    for s in ["forward_euler", "adams_bashforth"]:
        for N in Ns:
            dt05_results[s][N] = {}
            for F in F_values:
                cfg_file = os.path.join(shell_dir, s, f"N_{N}", "data", f"shell_N_{N}_F_{int(F)}.single_body.config")
                if os.path.exists(cfg_file):
                    with open(cfg_file, "r") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    idx = 0
                    frames = []
                    while idx < len(lines):
                        nb = int(lines[idx].split()[0])
                        idx += 1
                        bodies = []
                        for _ in range(nb):
                            bodies.append([float(x) for x in lines[idx].split()])
                            idx += 1
                        frames.append(bodies)
                    vz05 = abs((frames[-1][0][2] - frames[0][0][2]) / ((len(frames) - 1) * 0.05))
                    dt05_results[s][N][F] = vz05
                    
    # Panel 3A: Difference |V(dt=0.01) - V(mobility)| across all N and F
    diffs_fe = [abs(results["forward_euler"][N][F] - mobility_results[N][F]) for N in Ns for F in F_values]
    diffs_ab = [abs(results["adams_bashforth"][N][F] - mobility_results[N][F]) for N in Ns for F in F_values]
    
    # Bar plot of deviations from mobility for F=1000
    x_indices = np.arange(len(Ns))
    width = 0.22
    
    fe_001_diff = [abs(results["forward_euler"][N][1000.0] - mobility_results[N][1000.0]) for N in Ns]
    ab_001_diff = [abs(results["adams_bashforth"][N][1000.0] - mobility_results[N][1000.0]) for N in Ns]
    
    # If dt=0.05 exists
    fe_05_diff = [abs(dt05_results["forward_euler"][N].get(1000.0, mobility_results[N][1000.0]) - mobility_results[N][1000.0]) for N in Ns]
    ab_05_diff = [abs(dt05_results["adams_bashforth"][N].get(1000.0, mobility_results[N][1000.0]) - mobility_results[N][1000.0]) for N in Ns]
    
    # We plot the absolute terminal velocities comparing dt=0.01 and dt=0.05 and mobility
    ax3_1.plot(Ns, [mobility_results[N][1000.0] for N in Ns], 'k-', label='Mobility Reference', linewidth=2.5)
    ax3_1.plot(Ns, [results["forward_euler"][N][1000.0] for N in Ns], 's--', color='#1f77b4', label=f'Euler (dt={dt})', markersize=7)
    ax3_1.plot(Ns, [results["adams_bashforth"][N][1000.0] for N in Ns], 'o:', color='#2ca02c', label=f'Adams-Bashforth (dt={dt})', markersize=7)
    if all(1000.0 in dt05_results["forward_euler"][N] for N in Ns):
        ax3_1.plot(Ns, [dt05_results["forward_euler"][N][1000.0] for N in Ns], 'x-.', color='#d62728', label='Euler (dt=0.05)', markersize=7)
    ax3_1.set_xscale('log')
    ax3_1.set_title('Terminal Velocity Comparison (F = 1000 N)', fontsize=12, fontweight='bold')
    ax3_1.set_xlabel('Resolution N', fontsize=11)
    ax3_1.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
    ax3_1.set_xticks(Ns)
    ax3_1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3_1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax3_1.legend(fontsize=9)
    
    # Panel 3B: Numerical Difference relative to Mobility (Machine epsilon level)
    # Add a tiny epsilon 1e-16 to avoid log(0)
    eps = 1e-16
    ax3_2.bar(x_indices - 1.5*width, [max(d, eps) for d in fe_001_diff], width, label=f'|Euler(dt={dt}) - Mob|', color='#1f77b4', alpha=0.85)
    ax3_2.bar(x_indices - 0.5*width, [max(d, eps) for d in ab_001_diff], width, label=f'|AB(dt={dt}) - Mob|', color='#2ca02c', alpha=0.85)
    ax3_2.bar(x_indices + 0.5*width, [max(d, eps) for d in fe_05_diff], width, label='|Euler(dt=0.05) - Mob|', color='#d62728', alpha=0.85)
    ax3_2.bar(x_indices + 1.5*width, [max(d, eps) for d in ab_05_diff], width, label='|AB(dt=0.05) - Mob|', color='#ff7f0e', alpha=0.85)
    
    ax3_2.set_yscale('log')
    ax3_2.set_xticks(x_indices)
    ax3_2.set_xticklabels([f"N={N}" for N in Ns])
    ax3_2.set_title('Absolute Residual Error vs Mobility Reference (|V - V_mob|)', fontsize=12, fontweight='bold')
    ax3_2.set_xlabel('Discretization Resolution', fontsize=11)
    ax3_2.set_ylabel('Absolute Residual |ΔV|', fontsize=11)
    ax3_2.axhline(1e-14, color='gray', linestyle=':', label='Machine Precision (~10⁻¹⁴)')
    ax3_2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax3_2.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    p3 = os.path.join(shell_dir, f"dt_{dt}_sensitivity_comparison.png")
    plt.savefig(p3, dpi=300)
    print(f"Plot 3 saved: {p3}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
