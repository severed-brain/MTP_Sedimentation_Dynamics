import os
import sys
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
multi_bodies_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
custom_dir = os.path.join(base_dir, "custom_simulations", "fig3_replication")
os.makedirs(custom_dir, exist_ok=True)
os.makedirs(os.path.join(custom_dir, "data"), exist_ok=True)

N = 162
b_over_a_vals = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
F = 2000.0
g_val = F / N

def create_ellipsoid(input_vertex, output_vertex, scale_x, scale_y, scale_z):
    with open(input_vertex, 'r') as f:
        lines = f.readlines()
    with open(output_vertex, 'w') as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                x = float(parts[0]) * scale_x
                y = float(parts[1]) * scale_y
                z = float(parts[2]) * scale_z
                f.write(f"{x:.15e}\t{y:.15e}\t{z:.15e}\n")

v_src = os.path.join(multi_bodies_dir, "Structures", f"shell_N_{N}_Rg_1_Rh_1_0530.vertex")

env = os.environ.copy()
env["PYTHONPATH"] = f"{multi_bodies_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

# To hold the results
vel_end_on = {}
vel_broad_side = {}

# We need a standard clones file (no rotation)
c_dst = os.path.join(custom_dir, "single_body.clones")
with open(c_dst, "w") as f:
    f.write("1\n0 0 0 1 0 0 0\n")

for b_over_a in b_over_a_vals:
    a = 1.0 / b_over_a
    # b = c = 1.0
    
    # Calculate surface area to adapt blob_radius
    if a == 1.0:
        area = 4 * math.pi
    else:
        e = math.sqrt(1.0 - (1.0 / a)**2)
        area = 2 * math.pi * (1.0 + (a / e) * math.asin(e))
    
    optimal_blob_radius = 0.5 * math.sqrt(area / N)
    
    # --- 1. End-On (Parallel to gravity) ---
    # Gravity is Z. Long axis is Z. So stretch Z by a.
    v_end_on = os.path.join(custom_dir, f"end_on_{b_over_a}.vertex")
    create_ellipsoid(v_src, v_end_on, 1.0, 1.0, a)
    
    out_end = f"data/end_on_{b_over_a}"
    inp_end = os.path.join(custom_dir, f"input_end_{b_over_a}.dat")
    inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  30
n_save                                   1
eta                                      1.0
g                                        {g_val}
blob_radius                              {optimal_blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_end}
structure end_on_{b_over_a}.vertex single_body.clones
"""
    with open(inp_end, "w") as f: f.write(inp_content)
    subprocess.run([sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_end_{b_over_a}.dat"], cwd=custom_dir, env=env, check=True, capture_output=True)
    
    # Parse
    with open(os.path.join(custom_dir, f"{out_end}.single_body.config"), "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    frames = []
    idx = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        frames.append(bodies)
    vz_end = abs((frames[-1][0][2] - frames[0][0][2]) / ((len(frames)-1) * 0.05))
    vel_end_on[b_over_a] = vz_end
    
    # --- 2. Broad-side On (Perpendicular to gravity) ---
    # Gravity is Z. Long axis is X. So stretch X by a.
    v_broad = os.path.join(custom_dir, f"broad_{b_over_a}.vertex")
    create_ellipsoid(v_src, v_broad, a, 1.0, 1.0)
    
    out_broad = f"data/broad_{b_over_a}"
    inp_broad = os.path.join(custom_dir, f"input_broad_{b_over_a}.dat")
    inp_content = f"""scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  30
n_save                                   1
eta                                      1.0
g                                        {g_val}
blob_radius                              {optimal_blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_broad}
structure broad_{b_over_a}.vertex single_body.clones
"""
    with open(inp_broad, "w") as f: f.write(inp_content)
    subprocess.run([sys.executable, os.path.join(multi_bodies_dir, "multi_bodies.py"), "--input-file", f"input_broad_{b_over_a}.dat"], cwd=custom_dir, env=env, check=True, capture_output=True)
    
    # Parse
    with open(os.path.join(custom_dir, f"{out_broad}.single_body.config"), "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    frames = []
    idx = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        frames.append(bodies)
    vz_broad = abs((frames[-1][0][2] - frames[0][0][2]) / ((len(frames)-1) * 0.05))
    vel_broad_side[b_over_a] = vz_broad
    
    print(f"b/a={b_over_a} | End-on: {vz_end:.5f} | Broad-side: {vz_broad:.5f}")


# Normalize by the sphere velocity (b/a = 1.0)
u_sphere = vel_end_on[1.0]

sim_b_over_a = sorted(b_over_a_vals)
sim_end_on_norm = [vel_end_on[ba]/u_sphere for ba in sim_b_over_a]
sim_broad_norm = [vel_broad_side[ba]/u_sphere for ba in sim_b_over_a]

# Analytical curves
theory_b_over_a = np.linspace(0.28, 1.0, 100)
theory_end_on = []
theory_broad = []

for ba in theory_b_over_a:
    if ba == 1.0:
        theory_end_on.append(1.0)
        theory_broad.append(1.0)
        continue
    e = np.sqrt(1 - ba**2)
    # C_parallel
    c_para = (8/3) * (e**3) / (-2*e + (1 + e**2)*np.log((1+e)/(1-e)))
    # C_perp
    c_perp = (16/3) * (e**3) / (2*e + (3*e**2 - 1)*np.log((1+e)/(1-e)))
    
    theory_end_on.append(ba / c_para)
    theory_broad.append(ba / c_perp)

plt.figure(figsize=(8, 6))
plt.plot(theory_b_over_a, theory_end_on, '-', color='green', label=r'Analytical, $\hat{e} \parallel \hat{g}$')
plt.plot(theory_b_over_a, theory_broad, '-', color='red', label=r'Analytical, $\hat{e} \perp \hat{g}$')

plt.plot(sim_b_over_a, sim_end_on_norm, 'o', color='green', label=r'Simulations, $\hat{e} \parallel \hat{g}$')
plt.plot(sim_b_over_a, sim_broad_norm, '*', color='red', label=r'Simulations, $\hat{e} \perp \hat{g}$')

plt.xlabel('b/a')
plt.ylabel('|U|, normalised')
plt.legend()
plt.title('Replication of Figure 3 (Sedimenting Spheroid)')

save_path = os.path.join(custom_dir, "fig3_replication.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
