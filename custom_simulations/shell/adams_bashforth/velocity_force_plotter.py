import os
import sys
import subprocess
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Adams-Bashforth velocity vs force plotter")
parser.add_argument("--rerun", action="store_true", help="Force re-running simulations")
args = parser.parse_args()

# Configuration
Ns = [12, 42, 162, 642, 2562]
# Instead of fixed g, we use fixed total forces so the x-axis aligns for all N
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "shell", "adams_bashforth")

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
        f.write("1\n0 0 0 1 0 0 0\n")   #x y z and quaterneons
        
    for F in F_values:  # loop over the input files for different n
        g = F / N
        out_name = f"data/shell_N_{N}_F_{int(F)}"
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        
        # Calculate optimal blob radius for this N
        # Surface area = 4 * pi * R^2. Distance between blobs ~ sqrt(Area / N).
        # Optimal blob radius is roughly half the distance between blobs.
        import math
        optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
        
        # Write input file
        steps = 5 if N >= 642 else 20
        inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
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
            
        cfg_file = os.path.join(n_dir, f"{out_name}.single_body.config")
        if args.rerun or not os.path.exists(cfg_file):
            cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_F_{int(F)}.dat"]
            print(f"[Adams-Bashforth] Running N={N}, F={F} (g={g:.3f})...", flush=True)
            subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
        else:
            print(f"[Adams-Bashforth] Using existing output for N={N}, F={F}", flush=True)

        # Parse velocity
        with open(cfg_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        idx = 0
        frames = []
        while idx < len(lines):  # line by line timestamps
            nb = int(lines[idx].split()[0])
            idx += 1
            bodies = []
            for _ in range(nb):
                bodies.append([float(x) for x in lines[idx].split()])
                idx += 1
            frames.append(bodies)
            
        dt = 0.05
        z0 = frames[0][0][2]
        z_end = frames[-1][0][2]
        num_frames = len(frames) - 1
        vz = abs((z_end - z0) / (num_frames * dt)) # absolute velocity for plotting
        
        total_force = N * g
        forces.append(total_force)
        velocities.append(vz)
        
    plt.plot(forces, velocities, marker='o', label=f'Simulation (N = {N})')

# Add Theoretical Stokes Law
import numpy as np
all_forces = np.linspace(0, max(F_values), 100)
# Stokes Law: V = F / (6 * pi * eta * Rh)
# Here eta = 1.0, Rh = 1.0
theoretical_v = all_forces / (6 * np.pi * 1.0 * 1.0)
plt.plot(all_forces, theoretical_v, '--', color='black', label='Theoretical (Stokes Law)')

plt.title('Terminal Velocity vs Total Force for Single Shells')
plt.xlabel('Total Force (N * g)')
plt.ylabel('Terminal Velocity (|V_z|)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(custom_dir, "velocity_vs_force.png"))
print("Plot saved to velocity_vs_force.png")
