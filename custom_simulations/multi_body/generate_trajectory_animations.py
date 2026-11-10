import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"d:\sedimentation_dynamics"

base_dir = get_base_dir()
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
data_dir = os.path.join(mb_custom, "data")
plots_dir = os.path.join(mb_custom, "plots")
anim_dir = os.path.join(plots_dir, "animations")
thesis_anim_dir = os.path.join(base_dir, "thesis", "figures", "animations")

os.makedirs(anim_dir, exist_ok=True)
os.makedirs(thesis_anim_dir, exist_ok=True)

# Load trajectories summary
summary_file = os.path.join(data_dir, "trajectories_summary.json")
with open(summary_file, "r") as f:
    traj_data = json.load(f)

# Helper to save images as animated GIF
def make_gif(frame_paths, out_path, duration=100):
    images = [Image.open(p) for p in frame_paths]
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    # Cleanup frame files
    for p in frame_paths:
        if os.path.exists(p):
            os.remove(p)

# -------------------------------------------------------------------------
# ANIMATION 1: SPHERE PAIR SEDIMENTATION DRAFTING
# -------------------------------------------------------------------------
def animate_sphere_pair():
    print("Generating Animation 1: Sphere Pair Hydrodynamic Drafting...")
    pair_sim = traj_data["sphere"]["pair"]["2.0"]
    single_sim = traj_data["sphere"]["single"]["0.0"]
    
    t_arr = np.array(pair_sim["times"])
    b1_arr = np.array(pair_sim["traj_b1"])
    b2_arr = np.array(pair_sim["traj_b2"])
    s_arr = np.array(single_sim["traj_b1"])
    
    n_frames = min(len(t_arr), 35)
    frame_indices = np.linspace(0, len(t_arr) - 1, n_frames, dtype=int)
    frame_files = []
    
    for idx, f_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(8, 9))
        
        # Plot full trajectory trails up to current time
        ax.plot(b1_arr[:f_idx+1, 0], b1_arr[:f_idx+1, 2], 'r-', linewidth=2.0, alpha=0.7)
        ax.plot(b2_arr[:f_idx+1, 0], b2_arr[:f_idx+1, 2], 'b-', linewidth=2.0, alpha=0.7)
        ax.plot(s_arr[:f_idx+1, 0] + 4.0, s_arr[:f_idx+1, 2], 'k:', linewidth=2.0, alpha=0.7)
        
        # Plot current particle spheres
        # Pair spheres at x=-1 and x=+1
        c1 = plt.Circle((b1_arr[f_idx, 0], b1_arr[f_idx, 2]), 0.8, color='red', ec='black', zorder=5, label='Pair Body 1')
        c2 = plt.Circle((b2_arr[f_idx, 0], b2_arr[f_idx, 2]), 0.8, color='blue', ec='black', zorder=5, label='Pair Body 2')
        # Isolated reference sphere shifted to x = 4.0
        c_s = plt.Circle((4.0, s_arr[f_idx, 2]), 0.8, color='gray', ec='black', zorder=5, label='Solitary Sphere (Isolated)')
        
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.add_patch(c_s)
        
        # Vertical drafting speedup indicator
        z_pair = (b1_arr[f_idx, 2] + b2_arr[f_idx, 2]) / 2.0
        z_single = s_arr[f_idx, 2]
        delta_z = abs(z_pair - z_single)
        
        ax.set_xlim(-4.0, 6.0)
        ax.set_ylim(-75.0, 5.0)
        ax.set_title(f"Sphere Pair Hydrodynamic Drafting ($d/R = 2.0$)\nTime $t = {t_arr[f_idx]:.2f}$ s | Pair Lead: $\\Delta z = {delta_z:.1f}$ units", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Lateral Position $x$", fontsize=11)
        ax.set_ylabel("Vertical Position $z$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc='upper left')
        
        plt.tight_layout()
        f_name = os.path.join(anim_dir, f"_temp_sph_{idx:03d}.png")
        plt.savefig(f_name, dpi=120)
        plt.close()
        frame_files.append(f_name)
        
    out_gif = os.path.join(anim_dir, "anim_sphere_pair_sedimentation.gif")
    make_gif(frame_files, out_gif, duration=80)
    import shutil
    shutil.copyfile(out_gif, os.path.join(thesis_anim_dir, "anim_sphere_pair_sedimentation.gif"))
    print(f"Saved: {out_gif}")

# -------------------------------------------------------------------------
# ANIMATION 2: ELLIPSOID ANGLE TRAJECTORIES (0°, 45°, 90°)
# -------------------------------------------------------------------------
def animate_ellipsoid_angles():
    print("Generating Animation 2: Ellipsoid Angle Trajectories (0°, 45°, 90°)...")
    sim_0 = traj_data["ellipsoid"]["ang_0"]["0.0"]
    sim_45 = traj_data["ellipsoid"]["ang_45"]["0.0"]
    sim_90 = traj_data["ellipsoid"]["ang_90"]["0.0"]
    
    t_arr = np.array(sim_0["times"])
    arr_0 = np.array(sim_0["traj_b1"])
    arr_45 = np.array(sim_45["traj_b1"])
    arr_90 = np.array(sim_90["traj_b1"])
    
    # Offset x so they don't collide visually: 0 at -5, 45 at 0, 90 at +5
    x_off_0 = -5.0
    x_off_45 = 0.0
    x_off_90 = 5.0
    
    n_frames = min(len(t_arr), 35)
    frame_indices = np.linspace(0, len(t_arr) - 1, n_frames, dtype=int)
    frame_files = []
    
    for idx, f_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Full trails
        ax.plot(arr_0[:f_idx+1, 0] + x_off_0, arr_0[:f_idx+1, 2], color="#1f77b4", linewidth=2.2, label="θ = 0° (Slender/Fast)")
        ax.plot(arr_45[:f_idx+1, 0] + x_off_45, arr_45[:f_idx+1, 2], color="#d62728", linewidth=2.2, label="θ = 45° (Oblique Glider)")
        ax.plot(arr_90[:f_idx+1, 0] + x_off_90, arr_90[:f_idx+1, 2], color="#2ca02c", linewidth=2.2, label="θ = 90° (Broadside/Slow)")
        
        # Current positions
        ax.plot(arr_0[f_idx, 0] + x_off_0, arr_0[f_idx, 2], 'o', color="#1f77b4", markersize=12, markeredgecolor='black')
        ax.plot(arr_45[f_idx, 0] + x_off_45, arr_45[f_idx, 2], '^', color="#d62728", markersize=14, markeredgecolor='black')
        ax.plot(arr_90[f_idx, 0] + x_off_90, arr_90[f_idx, 2], 's', color="#2ca02c", markersize=12, markeredgecolor='black')
        
        # Annotation for lateral drift on 45 deg
        drift_x = arr_45[f_idx, 0] - arr_45[0, 0]
        ax.annotate(f"Lateral Drift:\n$\\Delta x = {drift_x:+.2f}$", 
                    (arr_45[f_idx, 0] + x_off_45 + 0.4, arr_45[f_idx, 2]),
                    fontsize=8, fontweight='bold', color="#d62728")
        
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(-60.0, 5.0)
        ax.set_title(f"Ellipsoid Sedimentation: Slender vs Oblique Gliding ($t = {t_arr[f_idx]:.2f}$ s)", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Lateral Position $x$", fontsize=11)
        ax.set_ylabel("Vertical Position $z$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc='lower left')
        
        plt.tight_layout()
        f_name = os.path.join(anim_dir, f"_temp_ell_{idx:03d}.png")
        plt.savefig(f_name, dpi=120)
        plt.close()
        frame_files.append(f_name)
        
    out_gif = os.path.join(anim_dir, "anim_ellipsoid_angles_trajectories.gif")
    make_gif(frame_files, out_gif, duration=80)
    import shutil
    shutil.copyfile(out_gif, os.path.join(thesis_anim_dir, "anim_ellipsoid_angles_trajectories.gif"))
    print(f"Saved: {out_gif}")

# -------------------------------------------------------------------------
# ANIMATION 3: BOOMERANG CHIRAL CORKSCREW TRAJECTORY
# -------------------------------------------------------------------------
def animate_boomerang_chiral():
    print("Generating Animation 3: Boomerang Chiral 3D Corkscrew Trajectory...")
    sim = traj_data["boomerang"]["single"]["0.0"]
    t_arr = np.array(sim["times"])
    arr = np.array(sim["traj_b1"])
    
    n_frames = min(len(t_arr), 35)
    frame_indices = np.linspace(0, len(t_arr) - 1, n_frames, dtype=int)
    frame_files = []
    
    from mpl_toolkits.mplot3d import Axes3D
    
    for idx, f_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 3D trajectory
        ax.plot(arr[:f_idx+1, 0], arr[:f_idx+1, 1], arr[:f_idx+1, 2], 'm-', linewidth=2.5, alpha=0.8)
        # Current particle marker
        ax.scatter([arr[f_idx, 0]], [arr[f_idx, 1]], [arr[f_idx, 2]], color='red', s=80, edgecolors='black')
        
        # Trajectory projections on bottom plane
        ax.plot(arr[:f_idx+1, 0], arr[:f_idx+1, 1], np.full(f_idx+1, -65.0), 'gray', linestyle=':', linewidth=1.2)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-65.0, 5.0)
        ax.set_title(f"Boomerang Chiral Corkscrew Trajectory ($t = {t_arr[f_idx]:.2f}$ s)\nCoupled Vertical Settling & Horizontal Precession", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Lateral $x$", fontsize=10)
        ax.set_ylabel("Lateral $y$", fontsize=10)
        ax.set_zlabel("Vertical $z$", fontsize=10)
        ax.view_init(elev=25, azim=45 + idx * 2.0)
        
        plt.tight_layout()
        f_name = os.path.join(anim_dir, f"_temp_boom_{idx:03d}.png")
        plt.savefig(f_name, dpi=120)
        plt.close()
        frame_files.append(f_name)
        
    out_gif = os.path.join(anim_dir, "anim_boomerang_chiral_trajectory.gif")
    make_gif(frame_files, out_gif, duration=80)
    import shutil
    shutil.copyfile(out_gif, os.path.join(thesis_anim_dir, "anim_boomerang_chiral_trajectory.gif"))
    print(f"Saved: {out_gif}")

# -------------------------------------------------------------------------
# ANIMATION 4: 5-BODY CROWLEY INSTABILITY (LINEAR CHAIN SAG)
# -------------------------------------------------------------------------
def animate_5body_chain():
    print("Generating Animation 4: 5-Sphere Crowley Instability Chain Sag...")
    # Read config from sphere 5-body chain at d/R = 3.0
    cfg_file = os.path.join(base_dir, "custom_simulations", "sphere", "multi_body", "data", "five_body", 
                            "linear_chain", "d_3_0", "sphere_5b_linear_chain_d_3_0.sphere_5b_linear_chain_d_3_0.config")
    if not os.path.exists(cfg_file):
        print(f"Warning: {cfg_file} not found. Skipping 5-body animation.")
        return
        
    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    all_trajs = [[] for _ in range(5)]
    times = []
    step = 0
    dt = 0.02
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        for b in range(nb):
            all_trajs[b].append([float(x) for x in lines[idx].split()])
            idx += 1
        times.append(step * dt)
        step += 1
        
    n_frames = min(len(times), 35)
    frame_indices = np.linspace(0, len(times) - 1, n_frames, dtype=int)
    frame_files = []
    
    body_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#ff7f0e", "#1f77b4"]
    
    for idx, f_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Plot chain connecting line at current step
        xs = [all_trajs[b][f_idx][0] for b in range(5)]
        zs = [all_trajs[b][f_idx][2] for b in range(5)]
        ax.plot(xs, zs, 'k--', linewidth=1.5, alpha=0.6)
        
        for b in range(5):
            # Trail
            tb = np.array(all_trajs[b])
            ax.plot(tb[:f_idx+1, 0], tb[:f_idx+1, 2], color=body_colors[b], linewidth=1.8, alpha=0.6)
            # Body circle
            circ = plt.Circle((xs[b], zs[b]), 0.9, color=body_colors[b], ec='black', zorder=5)
            ax.add_patch(circ)
            ax.text(xs[b], zs[b], f"{b+1}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
        # Sagging factor annotation
        center_lead = abs(zs[2] - ((zs[0] + zs[4]) / 2.0))
        
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(-110.0, 5.0)
        ax.set_title(f"5-Body Crowley Instability: Hydrodynamic Chain Sag ($d/R = 3.0$)\n$t = {times[f_idx]:.2f}$ s | Center Body Sag: $\\Delta z = {center_lead:.2f}$ units", 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Lateral Position $x$", fontsize=10)
        ax.set_ylabel("Vertical Position $z$", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        f_name = os.path.join(anim_dir, f"_temp_5b_{idx:03d}.png")
        plt.savefig(f_name, dpi=120)
        plt.close()
        frame_files.append(f_name)
        
    out_gif = os.path.join(anim_dir, "anim_5body_chain_crowley_instability.gif")
    make_gif(frame_files, out_gif, duration=80)
    import shutil
    shutil.copyfile(out_gif, os.path.join(thesis_anim_dir, "anim_5body_chain_crowley_instability.gif"))
    print(f"Saved: {out_gif}")

def main():
    print("=" * 95)
    print("GENERATING TRAJECTORY ANIMATIONS (.GIF) FOR MULTI-BODY SEDIMENTATION")
    print("=" * 95)
    animate_sphere_pair()
    animate_ellipsoid_angles()
    animate_boomerang_chiral()
    animate_5body_chain()
    print("=" * 95)
    print("ALL TRAJECTORY ANIMATIONS GENERATED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()
