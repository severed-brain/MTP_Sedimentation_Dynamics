import os
import sys
import math
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
struct_dir = os.path.join(mb_custom, "structures")
data_dir = os.path.join(mb_custom, "data", "helix")
plots_dir = os.path.join(mb_custom, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

N = 38
blob_radius = 0.1254
F = 1000.0
g = F / N
dt = 0.02
n_steps = 50
T = dt * n_steps
R_char = 1.0 # characteristic interaction length scale
d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]

v_file = os.path.join(struct_dir, "helix_N_38.vertex")
if not os.path.exists(v_file):
    import shutil
    src = os.path.join(base_dir, "custom_simulations", "helix", "single", "structures", "helix_N_38.vertex")
    shutil.copyfile(src, v_file)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

print("=" * 80)
print("MULTI-BODY PAIR SEDIMENTATION: HELIX STRUCTURE")
print(f"Discretization: N={N}, Blob Radius: a={blob_radius:.4f}, dt={dt}, Steps={n_steps}")
print("=" * 80)

# 1. Single Body Baseline
single_out = "helix_single"
single_clones = os.path.join(data_dir, f"{single_out}.clones")
single_inp = os.path.join(data_dir, f"input_{single_out}.dat")
single_cfg = os.path.join(data_dir, f"{single_out}.{single_out}.config")

with open(single_clones, "w") as f:
    f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")

inp_content = f"""scheme                                   deterministic_forward_euler
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        {g:.6f}
blob_radius                              {blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {single_out}
structure {v_file} {single_clones}
"""
with open(single_inp, "w") as f:
    f.write(inp_content)

print("Running baseline isolated single helix...", flush=True)
cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", f"input_{single_out}.dat"]
subprocess.run(cmd, cwd=data_dir, env=env, check=True, capture_output=True)

with open(single_cfg, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

idx = 0
times_single = []
traj_single = []
step = 0
while idx < len(lines):
    nb = int(lines[idx].split()[0])
    idx += 1
    bodies = []
    for _ in range(nb):
        bodies.append([float(x) for x in lines[idx].split()])
        idx += 1
    traj_single.append(bodies[0])
    times_single.append(step * dt)
    step += 1

v0_single = abs((traj_single[-1][2] - traj_single[0][2]) / T)
print(f"Isolated single helix terminal velocity: V0 = {v0_single:.4f} units/s")

# 2. Pair Simulations across d/R
pair_results = {}

for d_r in d_ratios:
    d = d_r * R_char
    x1 = -d / 2.0
    x2 = +d / 2.0
    out_base = f"helix_pair_d_{int(d_r)}"
    pair_clones = os.path.join(data_dir, f"{out_base}.clones")
    pair_inp = os.path.join(data_dir, f"input_{out_base}.dat")
    pair_cfg = os.path.join(data_dir, f"{out_base}.{out_base}.config")
    
    with open(pair_clones, "w") as f:
        f.write(f"2\n{x1:.6f} 0.0 0.0 1.0 0.0 0.0 0.0\n{x2:.6f} 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    inp_content = f"""scheme                                   deterministic_forward_euler
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        {g:.6f}
blob_radius                              {blob_radius:.4f}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_base}
structure {v_file} {pair_clones}
"""
    with open(pair_inp, "w") as f:
        f.write(inp_content)
        
    print(f"Running Helix pair simulation d/R = {d_r:.1f}...", flush=True)
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", f"input_{out_base}.dat"]
    subprocess.run(cmd, cwd=data_dir, env=env, check=True, capture_output=True)
    
    with open(pair_cfg, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    t_b1, t_b2 = [], []
    times = []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        t_b1.append(bodies[0])
        t_b2.append(bodies[1])
        times.append(step * dt)
        step += 1
        
    vz_pair = abs((t_b1[-1][2] - t_b1[0][2]) / T)
    enhancement = vz_pair / v0_single
    pair_results[d_r] = {
        "d_r": d_r,
        "times": times,
        "traj_b1": t_b1,
        "traj_b2": t_b2,
        "vz_pair": vz_pair,
        "enhancement": enhancement
    }
    print(f"  d/R = {d_r:4.1f}: Vz = {vz_pair:7.3f} units/s, Enhancement = {enhancement:.4f}")

# Save JSON summary
summary_file = os.path.join(data_dir, "helix_pair_summary.json")
with open(summary_file, "w") as f:
    json.dump({
        "v0_single": v0_single,
        "d_ratios": d_ratios,
        "enhancements": [pair_results[d]["enhancement"] for d in d_ratios],
        "vz_pairs": [pair_results[d]["vz_pair"] for d in d_ratios]
    }, f, indent=2)

# Generate 2-Panel Figure
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors = {2.0: "#d62728", 4.0: "#ff7f0e", 6.0: "#2ca02c", 8.0: "#1f77b4", 10.0: "#9467bd"}

# Panel A: Vertical Trajectories
ax1 = axes[0]
ax1.plot(times_single, [p[2] for p in traj_single], "k--", linewidth=2.5, label=f"Isolated Single Helix ($V_0={v0_single:.2f}$)")
for d_r in d_ratios:
    res = pair_results[d_r]
    ax1.plot(res["times"], [p[2] for p in res["traj_b1"]], "-", color=colors[d_r], linewidth=2.0, 
             label=f"Pair $d/R={int(d_r)}$ ($V_z={res['vz_pair']:.2f}$)")
ax1.set_title("Helix Pair Sedimentation Trajectories $z(t)$", fontsize=13, fontweight="bold")
ax1.set_xlabel("Time $t$ (s)", fontsize=11)
ax1.set_ylabel("Vertical Position $z$ (units)", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=9, loc="lower left")

# Panel B: Velocity Enhancement vs Separation Distance d/R
ax2 = axes[1]
enh_vals = [pair_results[d]["enhancement"] for d in d_ratios]
ax2.plot(d_ratios, enh_vals, "ro-", linewidth=2.2, markersize=8, label="Multiblob Simulation $V_z(d)/V_0$")

# Theoretical Stokeslet comparison: 1 + 3/4 * (R_eff / d)
# Fit R_eff from the far-field:
d_arr = np.array(d_ratios)
r_eff_fit = np.polyfit(1.0 / d_arr, np.array(enh_vals) - 1.0, 1)[0] / 0.75
theory_curve = 1.0 + 0.75 * r_eff_fit / d_arr
ax2.plot(d_ratios, theory_curve, "b--", linewidth=1.8, label=f"Stokeslet Asymptotic Decay ($1 + \\frac{{3}}{{4}}\\frac{{R_{{eff}}}}{{d}}$)")

ax2.set_title("Cooperative Hydrodynamic Drafting Enhancement", fontsize=13, fontweight="bold")
ax2.set_xlabel("Separation Ratio $d/R$", fontsize=11)
ax2.set_ylabel("Velocity Enhancement $V_z(d) / V_0$", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.set_xticks(d_ratios)
ax2.legend(fontsize=10)

plt.tight_layout()

p1 = os.path.join(plots_dir, "helix_pair_trajectories.png")
p2 = os.path.join(thesis_fig_dir, "helix_pair_trajectories.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nGenerated and saved Helix Pair Trajectory plots:")
print(f"  -> {p1}")
print(f"  -> {p2}")

print("\n" + "=" * 80)
print(f"{'Separation d/R':<18} {'Pair Velocity Vz':<20} {'Enhancement Vz/V0':<20} {'Stokeslet Theory':<20}")
print("-" * 80)
for d_r in d_ratios:
    th = 1.0 + 0.75 * r_eff_fit / d_r
    print(f"{d_r:<18.1f} {pair_results[d_r]['vz_pair']:<20.3f} {pair_results[d_r]['enhancement']:<20.4f} {th:<20.4f}")
print("=" * 80)
