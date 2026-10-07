import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import math

# Configuration
Ns = [12, 42, 162, 642, 2562]
base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "shell")

vertex_files = {
    12: "shell_N_12_Rg_1_Rh_1_2625.vertex",
    42: "shell_N_42_Rg_1_Rh_1_1220.vertex",
    162: "shell_N_162_Rg_1_Rh_1_0530.vertex",
    642: "shell_N_642_Rg_1_Rh_1_0239.vertex",
    2562: "shell_N_2562_Rg_1_Rh_1_0113.vertex"
}

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

execution_times = []

# We will run exactly 5 time steps for every N to make it a perfectly fair comparison
steps = 5 
F = 1000.0

for N in Ns:
    n_dir = os.path.join(custom_dir, f"N_{N}")
    os.makedirs(n_dir, exist_ok=True)
    os.makedirs(os.path.join(n_dir, "data"), exist_ok=True)
    
    v_src = os.path.join(multi_bodies_dir, "Structures", vertex_files[N])
    v_dst = os.path.join(n_dir, vertex_files[N])
    if not os.path.exists(v_dst):
        import shutil
        shutil.copy(v_src, v_dst)
        
    c_dst = os.path.join(n_dir, "single_body.clones")
    with open(c_dst, "w") as f:
        f.write("1\n0 0 0 1 0 0 0\n")
        
    g = F / N
    out_name = f"data/shell_N_{N}_time_test"
    inp_file = os.path.join(n_dir, f"input_time_test.dat")
    
    optimal_blob_radius = 0.5 * math.sqrt(4 * math.pi / N)
    
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
        
    cmd = [sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", "input_time_test.dat"]
    print(f"Timing simulation for N={N} (5 steps)...")
    
    start_time = time.time()
    subprocess.run(cmd, cwd=n_dir, env=env, check=True, capture_output=True)
    end_time = time.time()
    
    elapsed = end_time - start_time
    execution_times.append(elapsed)
    print(f"N={N} took {elapsed:.2f} seconds")

# Plotting on a Linear Scale to show the exponential jump
plt.figure(figsize=(10, 6))
plt.plot(Ns, execution_times, marker='o', linestyle='-', color='red', markersize=8)
plt.xscale('linear')
plt.yscale('linear')

# Formatting the ticks so it reads normally instead of scientific notation
plt.xticks(Ns, labels=[str(n) for n in Ns])
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())

plt.title('Simulation Execution Time vs Number of Blobs (N)')
plt.xlabel('Number of Blobs (N)')
plt.ylabel('Execution Time for 5 Time Steps (seconds)')
plt.grid(True, which="both", ls="--", alpha=0.7)

save_path = os.path.join(custom_dir, "time_vs_N_plot.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
