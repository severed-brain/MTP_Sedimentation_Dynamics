import os
import sys
import subprocess
import math
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "ellipsoid_angles")
os.makedirs(custom_dir, exist_ok=True)
os.makedirs(os.path.join(custom_dir, "data"), exist_ok=True)

N = 162

# Create the prolate vertex file
def create_prolate_ellipsoid(input_vertex, output_vertex):
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

v_src = os.path.join(multi_bodies_dir, "Structures", f"shell_N_{N}_Rg_1_Rh_1_0530.vertex")
v_dst = os.path.join(custom_dir, f"ellipsoid_prolate_N_{N}.vertex")
create_prolate_ellipsoid(v_src, v_dst)

angles_deg = [0, 15, 30, 45, 60, 75, 90]
F = 2000.0
g = F / N
optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

plt.figure(figsize=(6, 8))

for alpha_deg in angles_deg:
    alpha_rad = math.radians(alpha_deg)
    
    # Quaternion for rotation around Y-axis by alpha
    # This tilts the long Z-axis towards the X-axis
    qw = math.cos(alpha_rad / 2.0)
    qx = 0.0
    qy = math.sin(alpha_rad / 2.0)
    qz = 0.0
    
    c_dst = os.path.join(custom_dir, f"clones_angle_{alpha_deg}.clones")
    with open(c_dst, "w") as f:
        f.write(f"1\n0 0 0 {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n")
        
    out_name = f"data/prolate_angle_{alpha_deg}"
    inp_file = os.path.join(custom_dir, f"input_angle_{alpha_deg}.dat")
    
    # We will simulate 100 steps to get a long enough trajectory to clearly see the drift
    inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  100
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
structure ellipsoid_prolate_N_{N}.vertex clones_angle_{alpha_deg}.clones
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_angle_{alpha_deg}.dat"]
    print(f"Running Angle = {alpha_deg} degrees...")
    subprocess.run(cmd, cwd=custom_dir, env=env, check=True, capture_output=True)
    
    # Parse Center of Mass trajectory
    cfg_file = os.path.join(custom_dir, f"{out_name}.clones_angle_{alpha_deg}.config")
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    com_x = []
    com_z = []
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        x_sum, z_sum = 0.0, 0.0
        for _ in range(nb):
            parts = lines[idx].split()
            x_sum += float(parts[0])
            z_sum += float(parts[2])
            idx += 1
        com_x.append(x_sum / nb)
        com_z.append(z_sum / nb)
        
    # Plot trajectory relative to origin
    # We want to subtract the initial position so all trajectories start at (0,0)
    com_x = [x - com_x[0] for x in com_x]
    com_z = [z - com_z[0] for z in com_z]
    
    # In Figure 5a of the paper, they plot Z on the Y axis, and X on the X axis. 
    # Z goes negative as it falls, which naturally plots downwards if we don't invert, 
    # but let's invert Y axis to match the visual style of falling downwards.
    
    plt.plot(com_x, com_z, marker='.', markersize=4, label=f'{alpha_deg}°')
    # Mark the final position with a larger dot
    plt.plot(com_x[-1], com_z[-1], marker='o', color='black', markersize=6)

plt.title('Prolate Ellipsoid Lateral Drift Trajectories (Fig 5a Replica)')
plt.xlabel('Lateral Drift (X coordinate)')
plt.ylabel('Vertical Sedimentation (Z coordinate)')

# DO NOT use plt.axis('equal') because the drift is an order of magnitude smaller than the vertical drop!
# (Exactly as mentioned in the paper's text for Figure 5a)

plt.legend(title="Angle (α)")
plt.grid(True, linestyle='--', alpha=0.6)
save_path = os.path.join(custom_dir, "lateral_drift_trajectories.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
