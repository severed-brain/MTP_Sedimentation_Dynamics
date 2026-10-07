import os
import sys
import argparse
import subprocess

base_dir = os.path.dirname(os.path.abspath(__file__))
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
data_dir = os.path.join(multi_bodies_dir, "data")
os.makedirs(data_dir, exist_ok=True)

files = {
    12: "Structures/shell_N_12_Rg_1_Rh_1_2625.vertex",
    42: "Structures/shell_N_42_Rg_1_Rh_1_1220.vertex",
    162: "Structures/shell_N_162_Rg_1_Rh_1_0530.vertex",
    642: "Structures/shell_N_642_Rg_1_Rh_1_0239.vertex",
    2562: "Structures/shell_N_2562_Rg_1_Rh_1_0113.vertex"
}

parser = argparse.ArgumentParser(description="Simulate sedimentation of rigid multiblob shells.")
parser.add_argument("--N", type=int, default=0, help="Preset shell blob count (12, 42, 162, 642, 2562)")
parser.add_argument("--steps", type=int, default=20, help="Number of integration steps")
parser.add_argument("--vertex", "--vertex-file", dest="vertex_file", type=str, default=None, help="Path to custom .vertex file")
parser.add_argument("--clones", "--clones-file", dest="clones_file", type=str, default=None, help="Path to custom .clones file")
parser.add_argument("--input-file", dest="input_file", type=str, default=None, help="Path to custom .dat input file")
args = parser.parse_args()

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir}{os.pathsep}{os.path.join(base_dir, 'RigidMultiblobsWall')}"

results = []

# Case 1: User provides custom vertex and clones files
if args.vertex_file is not None or args.input_file is not None:
    if args.input_file is not None:
        inp_file = os.path.abspath(args.input_file)
        cmd = [sys.executable, "multi_bodies.py", "--input-file", inp_file]
        subprocess.run(cmd, cwd=multi_bodies_dir, env=env, check=True)
        print(f"\nSimulation finished using input file: {inp_file}")
        sys.exit(0)
    else:
        v_file_path = os.path.abspath(args.vertex_file)
        c_file_path = os.path.abspath(args.clones_file) if args.clones_file else os.path.join(multi_bodies_dir, "Structures", "shell_N_12_Rg_1.clones")
        
        # Read N from vertex file
        with open(v_file_path, "r") as vf:
            first_line = vf.readline().strip()
            N = int(first_line.split()[0])
            
        inp_file = os.path.join(multi_bodies_dir, f"inputfile_custom_N_{N}.dat")
        out_name = f"data/custom_N_{N}_sedimentation"
        
        # Get structure ID from clones filename
        c_basename = os.path.basename(c_file_path)
        c_id = c_basename[:-7] if c_basename.endswith(".clones") else c_basename
        
        inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  {args.steps}
n_save                                   1
eta                                      1.0
g                                        1.0
blob_radius                              0.25
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_name}
structure {v_file_path} {c_file_path}
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        cmd = [sys.executable, "multi_bodies.py", "--input-file", os.path.basename(inp_file)]
        subprocess.run(cmd, cwd=multi_bodies_dir, env=env, check=True)
        
        cfg_file = os.path.join(multi_bodies_dir, f"{out_name}.{c_id}.config")
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
        z1 = frames[1][0][2]
        vz = (z1 - z0) / dt
        vz_norm = vz / N
        print("\nResults for custom user files:")
        print(f"Vertex File: {args.vertex_file}")
        print(f"Clones File: {c_file_path}")
        print(f"{'N':>6} | {'Steps':>6} | {'V_z (unit/s)':>14} | {'V_z / (N*g)':>14}")
        print("-" * 50)
        print(f"{N:6d} | {len(frames) - 1:6d} | {vz:14.5f} | {vz_norm:14.6f}")
        sys.exit(0)

# Case 2: Standard preset sweep (12, 42, 162, 642) or specific --N
N_list = [args.N] if args.N in files else [12, 42, 162, 642]

for N in N_list:
    v_file = files[N]
    inp_file = os.path.join(multi_bodies_dir, f"inputfile_shell_N_{N}_free_fall.dat")
    out_name = f"data/shell_N_{N}_free_fall"
    n_steps = 5 if N == 2562 else args.steps
    
    inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        1.0
blob_radius                              0.25
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_name}
structure {v_file} Structures/shell_N_12_Rg_1.clones
"""
    with open(inp_file, "w") as f:
        f.write(inp_content)
    
    cmd = [sys.executable, "multi_bodies.py", "--input-file", f"inputfile_shell_N_{N}_free_fall.dat"]
    subprocess.run(cmd, cwd=multi_bodies_dir, env=env, check=True)
    
    cfg_file = os.path.join(multi_bodies_dir, f"{out_name}.shell_N_12_Rg_1.config")
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
    z1 = frames[1][0][2]
    vz = (z1 - z0) / dt
    vz_norm = vz / N
    results.append((N, len(frames) - 1, vz, vz_norm))

print("\nConvergence Across Shell Discretizations (Free Fall):")
print(f"{'N':>6} | {'Steps':>6} | {'V_z (unit/s)':>14} | {'V_z / (N*g)':>14}")
print("-" * 50)
for r in results:
    print(f"{r[0]:6d} | {r[1]:6d} | {r[2]:14.5f} | {r[3]:14.6f}")
