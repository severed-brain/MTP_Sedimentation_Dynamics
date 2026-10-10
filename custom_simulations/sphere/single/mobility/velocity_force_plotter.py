import os
import sys
import subprocess
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Mobility Scheme velocity vs force plotter")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

# Configuration
Ns = [12, 42, 162, 642, 2562]
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "shell", "mobility")

vertex_files = {
    12: "shell_N_12_Rg_1_Rh_1_2625.vertex",
    42: "shell_N_42_Rg_1_Rh_1_1220.vertex",
    162: "shell_N_162_Rg_1_Rh_1_0530.vertex",
    642: "shell_N_642_Rg_1_Rh_1_0239.vertex",
    2562: "shell_N_2562_Rg_1_Rh_1_0113.vertex"
}

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

plt.figure(figsize=(10, 6))

for N in Ns:
    forces = []
    velocities = []
    
    # Create directory for this N
    n_dir = os.path.join(custom_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    # Copy vertex file if not exists
    v_src = os.path.join(multi_bodies_dir, "Structures", vertex_files[N])
    v_dst = os.path.join(n_dir, vertex_files[N])
    if not os.path.exists(v_dst):
        import shutil
        shutil.copy(v_src, v_dst)
        
    # Create single body clones file
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0 0 0 1 0 0 0\n")
        
    for F in F_values:
        g = F / N
        out_name = f"data/shell_N_{N}_F_{int(F)}"
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        vel_file = os.path.join(n_dir, f"{out_name}.velocity.dat")
        
        # Calculate optimal blob radius for this N
        optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
        
        # Write input file for mobility scheme
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
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        # Run simulation using multi_bodies_utilities.py if not cached or --rerun
        if args.rerun or not os.path.exists(vel_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies_utilities.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Mobility Scheme] Running N={N}, F={F} (g={g:.3f})...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
        else:
            print(f"[Mobility Scheme] Using existing output for N={N}, F={F}", flush=True)
        
        # Parse velocity from .velocity.dat
        vel_data = np.loadtxt(vel_file)
        if vel_data.ndim == 1:
            vz = abs(vel_data[2])
        else:
            vz = abs(vel_data[0, 2])
            
        total_force = N * g
        forces.append(total_force)
        velocities.append(vz)
        print(f"  -> Total Force: {total_force:.1f}, Terminal Velocity: {vz:.4f}")
        
    plt.plot(forces, velocities, marker='o', label=f'Mobility Scheme (N = {N})')

# Add Theoretical Stokes Law
all_forces = np.linspace(0, max(F_values), 100)
theoretical_v = all_forces / (6 * np.pi * 1.0 * 1.0)
plt.plot(all_forces, theoretical_v, '--', color='black', label='Theoretical (Stokes Law)')

plt.title('Terminal Velocity vs Total Force (Mobility Scheme)')
plt.xlabel('Total Force (N * g)')
plt.ylabel('Terminal Velocity (|V_z|)')
plt.legend()
plt.grid(True)
plot_path = os.path.join(custom_dir, "velocity_vs_force.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
