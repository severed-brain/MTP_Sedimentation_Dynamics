import os
import sys
import subprocess
import matplotlib.pyplot as plt
import math

# Configuration
Ns = [12, 42, 162, 642, 2562]
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "ellipsoid")

# Mapping from N to the base sphere vertex file
vertex_files = {
    12: "shell_N_12_Rg_1_Rh_1_2625.vertex",
    42: "shell_N_42_Rg_1_Rh_1_1220.vertex",
    162: "shell_N_162_Rg_1_Rh_1_0530.vertex",
    642: "shell_N_642_Rg_1_Rh_1_0239.vertex",
    2562: "shell_N_2562_Rg_1_Rh_1_0113.vertex"
}

def create_prolate_ellipsoid(input_vertex, output_vertex):
    """Stretches a sphere into a 1x1x2 prolate ellipsoid"""
    with open(input_vertex, 'r') as f:
        lines = f.readlines()
    with open(output_vertex, 'w') as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                x = float(parts[0]) * 1.0
                y = float(parts[1]) * 1.0
                z = float(parts[2]) * 2.0
                f.write(f"{x:.15e}\t{y:.15e}\t{z:.15e}\n")

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

plt.figure(figsize=(10, 6))

for N in Ns:
    n_dir = os.path.join(custom_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    # Generate the stretched Prolate Ellipsoid vertex file for this N
    v_src = os.path.join(multi_bodies_dir, "Structures", vertex_files[N])
    v_dst = os.path.join(n_dir, f"ellipsoid_prolate_N_{N}.vertex")
    create_prolate_ellipsoid(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0 0 0 1 0 0 0\n")
        
    forces = []
    velocities = []
    
    for F in F_values:
        g = F / N
        out_name = f"data/prolate_N_{N}_F_{int(F)}"
        inp_file = os.path.join(n_dir, f"input_F_{int(F)}.dat")
        
        # Optimal blob radius scales with 1/sqrt(N) just like before
        optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
        
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
structure ellipsoid_prolate_N_{N}.vertex single_body.clones
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_F_{int(F)}.dat"]
        print(f"Running Prolate Ellipsoid N={N}, F={F} (g={g:.3f})...")
        subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
        
        # Parse velocity
        cfg_file = os.path.join(n_dir, f"{out_name}.single_body.config")
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
            
        dt = 0.05
        z0 = frames[0][0][2]
        z_end = frames[-1][0][2]
        num_frames = len(frames) - 1
        vz = abs((z_end - z0) / (num_frames * dt))
        
        forces.append(F)
        velocities.append(vz)
        
    plt.plot(forces, velocities, marker='o', label=f'Simulation (N = {N})')

plt.title('Terminal Velocity vs Total Force for Prolate Ellipsoids (1x1x2)')
plt.xlabel('Total Force (N * g)')
plt.ylabel('Terminal Velocity (|V_z|)')
plt.legend()
plt.grid(True)
save_path = os.path.join(custom_dir, "ellipsoid_velocity_vs_force.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
