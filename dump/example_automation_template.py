import os
import sys
import subprocess
import math

# ---------------------------------------------------------
# AI AUTOMATION TEMPLATE FOR ARBITRARY SHAPES
# ---------------------------------------------------------
# This script is a template that an AI agent can use to 
# automate the generation and simulation of any arbitrary 
# shape in the RigidMultiblobsWall framework.
# ---------------------------------------------------------

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")

# 1. Define where the data will live
custom_dir = os.path.join(base_dir, "dump", "my_custom_shape_test")
os.makedirs(custom_dir, exist_ok=True)
os.makedirs(os.path.join(custom_dir, "data"), exist_ok=True) # CRITICAL: solver needs this to exist!

# 2. Geometry Generation
N = 162
v_src = os.path.join(multi_bodies_dir, "Structures", f"shell_N_{N}_Rg_1_Rh_1_0530.vertex")
v_dst = os.path.join(custom_dir, f"my_shape_N_{N}.vertex")

def generate_custom_shape(input_vertex, output_vertex):
    """
    AI TO DO: Implement logic to read original coordinates and 
    apply mathematical transformations (stretching, twisting, etc.)
    to create the new arbitrary shape.
    """
    with open(input_vertex, 'r') as f:
        lines = f.readlines()
    with open(output_vertex, 'w') as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                # Example: Stretch X by 2, Y by 0.5, Z by 3
                new_x, new_y, new_z = x * 2.0, y * 0.5, z * 3.0
                f.write(f"{new_x:.15e}\t{new_y:.15e}\t{new_z:.15e}\n")

generate_custom_shape(v_src, v_dst)

# 3. Orientation Generation
c_dst = os.path.join(custom_dir, "single_body.clones")
with open(c_dst, "w") as f:
    # 1 body, X=0, Y=0, Z=0, qw=1, qx=0, qy=0, qz=0
    f.write("1\n0 0 0 1 0 0 0\n")

# 4. Input Configuration
F = 2000.0
g = F / N

# AI TO DO: Calculate exact surface area of the new shape to find optimal blob radius
# fallback approximation for a sphere-like object:
optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N) 

out_name = "data/my_shape_output"
inp_file = os.path.join(custom_dir, "input.dat")

inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  50
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
structure my_shape_N_{N}.vertex single_body.clones
"""

with open(inp_file, "w") as f:
    f.write(inp_content)

# 5. Execution
env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", "input.dat"]
print(f"Running custom shape simulation...")

# CRITICAL: cwd must be set to custom_dir where .dat, .vertex, and .clones live!
subprocess.run(cmd, cwd=custom_dir, env=env, check=True, capture_output=True)

# 6. Data Parsing
cfg_file = os.path.join(custom_dir, f"{out_name}.single_body.config")
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
vz = abs((z_end - z0) / ((len(frames) - 1) * 0.05))

print(f"Simulation Complete. Terminal Velocity (Vz) = {vz:.5f}")
