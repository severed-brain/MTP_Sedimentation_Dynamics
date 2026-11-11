import os
import sys
import math
import json
import csv
import shutil
import numpy as np
import matplotlib.pyplot as plt

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
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

# Load trajectories summary
summary_file = os.path.join(data_dir, "trajectories_summary.json")
with open(summary_file, "r") as f:
    data = json.load(f)

# Load highres asymmetric pairs CSV
csv_path = os.path.join(data_dir, "highres_angle_pairs_ellipsoid.csv")
ell_csv_records = []
if os.path.exists(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ell_csv_records.append(row)

def main():
    print("=" * 95)
    print("PLOTTING DEDICATED ELLIPSOID TRAJECTORIES (θ = 0°, 45°, 90°)")
    print("Analyzing Solitary Oblique Gliding, Symmetrical Drafting, and Asymmetric Pair Interactions")
    print("=" * 95)
    
    angles = [0, 45, 90]
    ang_colors = {0: "#1f77b4", 45: "#d62728", 90: "#2ca02c"}
    ang_labels = {0: "θ = 0° (Vertical/Slender)", 45: "θ = 45° (Oblique Glider)", 90: "θ = 90° (Broadside/High Drag)"}
    d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # -------------------------------------------------------------------------
    # Panel A: Spatial Trajectories (x vs z) for Solitary vs Pair (d/R = 4)
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    # Single body trajectories
    for ang in angles:
        sim_single = data["ellipsoid"][f"ang_{ang}"]["0.0"]
        tb = np.array(sim_single["traj_b1"])
        ax_a.plot(tb[:, 0], tb[:, 2], ':', color=ang_colors[ang], linewidth=2.5, label=f"Single {ang_labels[ang]}")
        
    # Pair trajectories at d/R = 4
    for ang in angles:
        sim_pair = data["ellipsoid"][f"ang_{ang}"]["4.0"]
        tb1 = np.array(sim_pair["traj_b1"])
        tb2 = np.array(sim_pair["traj_b2"])
        ax_a.plot(tb1[:, 0], tb1[:, 2], '-', color=ang_colors[ang], linewidth=2.0, label=f"Pair Body 1 (θ={ang}°)")
        ax_a.plot(tb2[:, 0], tb2[:, 2], '--', color=ang_colors[ang], linewidth=2.0, label=f"Pair Body 2 (θ={ang}°)")
        
    ax_a.set_title(r"(A) Spatial Trajectories ($x$ vs $z$): Lateral Glide of $\theta=45^\circ$", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Lateral Position $x$", fontsize=11)
    ax_a.set_ylabel("Vertical Position $z$", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8, loc="lower center", ncol=2)
    
    # -------------------------------------------------------------------------
    # Panel B: Vertical Trajectory z(t) vs Time (Single Body Baselines)
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    for ang in angles:
        sim_single = data["ellipsoid"][f"ang_{ang}"]["0.0"]
        t = sim_single["times"]
        tb = np.array(sim_single["traj_b1"])
        vz = abs(tb[-1, 2] - tb[0, 2]) / (t[-1] - t[0])
        ax_b.plot(t, tb[:, 2], '-', color=ang_colors[ang], linewidth=2.5, label=f"Single θ={ang}°: $V_0 = {vz:.2f}$ u/s")
        
        # Also plot pair at d/R = 2.0 to show drafting speedup
        sim_p2 = data["ellipsoid"][f"ang_{ang}"]["2.0"]
        t_p = sim_p2["times"]
        tb_p = np.array(sim_p2["traj_b1"])
        vz_p = abs(tb_p[-1, 2] - tb_p[0, 2]) / (t_p[-1] - t_p[0])
        ax_b.plot(t_p, tb_p[:, 2], '--', color=ang_colors[ang], linewidth=1.8, alpha=0.8, label=f"Pair θ={ang}° ($d=2R$): $V_z = {vz_p:.2f}$ u/s")
        
    ax_b.set_title(r"(B) Vertical Sedimentation $z(t)$ vs Time: Baseline $V_0(\theta)$", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Time $t$ (s)", fontsize=11)
    ax_b.set_ylabel("Vertical Position $z$", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=8)
    
    # -------------------------------------------------------------------------
    # Panel C: Normalized Velocity Enhancement V_z / V_0(theta) vs d/R
    # -------------------------------------------------------------------------
    ax_c = axes[1, 0]
    for ang in angles:
        sim_single = data["ellipsoid"][f"ang_{ang}"]["0.0"]
        v0 = abs(sim_single["traj_b1"][-1][2] - sim_single["traj_b1"][0][2]) / (sim_single["times"][-1] - sim_single["times"][0])
        
        enh_ratios = []
        for d_r in d_ratios:
            sim = data["ellipsoid"][f"ang_{ang}"][str(d_r)]
            vz = abs(sim["traj_b1"][-1][2] - sim["traj_b1"][0][2]) / (sim["times"][-1] - sim["times"][0])
            enh_ratios.append(vz / v0)
            
        ax_c.plot(d_ratios, enh_ratios, 'o-', color=ang_colors[ang], linewidth=2.2, markersize=8, label=f"θ = {ang}° (Baseline $V_0 = {v0:.2f}$)")
        
    d_th = np.linspace(1.8, 10.5, 100)
    ax_c.plot(d_th, 1.0 + 0.75 * (1.0 / d_th), 'k--', linewidth=1.8, label=r'Stokeslet Asymptotic ($1 + \frac{3}{4}\frac{R}{d}$)')
    ax_c.axhline(1.0, color='gray', linestyle=':', label='Isolated Particle Baseline (1.0)')
    
    ax_c.set_title(r"(C) Normalized Hydrodynamic Drafting $V_z / V_0(\theta)$ vs $d/R$", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax_c.set_ylabel(r"Normalized Velocity Enhancement $V_z / V_0$", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(fontsize=8, loc="upper right")
    
    # -------------------------------------------------------------------------
    # Panel D: Differential Settling ΔVz in Asymmetric Pairs
    # -------------------------------------------------------------------------
    ax_d = axes[1, 1]
    if ell_csv_records:
        pair_styles = {
            "mismatched_0_45": ("θ₁=0°, θ₂=45°", "#1f77b4", "o-"),
            "mismatched_45_90": ("θ₁=45°, θ₂=90°", "#d62728", "s-"),
            "mismatched_0_90": ("θ₁=0°, θ₂=90°", "#2ca02c", "^-"),
            "symmetric_0_0": ("θ₁=0°, θ₂=0°", "#9467bd", "--"),
            "symmetric_45_45": ("θ₁=45°, θ₂=45°", "#ff7f0e", "--"),
            "symmetric_90_90": ("θ₁=90°, θ₂=90°", "#8c564b", "--")
        }
        for p_tag, (p_lbl, p_col, p_ls) in pair_styles.items():
            sub = [r for r in ell_csv_records if r["pair_type"] == p_tag]
            if sub:
                ds = [float(r["d_r"]) for r in sub]
                delta_v = [float(r["delta_Vz"]) for r in sub]
                ax_d.plot(ds, delta_v, p_ls, color=p_col, linewidth=2.0, markersize=7, label=p_lbl)
                
    ax_d.axhline(0.0, color='black', linewidth=1.0)
    ax_d.set_title(r"(D) Asymmetric Pair Differential Settling $\Delta V_z = |V_z^{(1)} - V_z^{(2)}|$", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax_d.set_ylabel(r"Differential Settling Velocity $\Delta V_z$ (units/s)", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.5)
    ax_d.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "ellipsoid_angle_trajectories_0_45_90.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated ellipsoid trajectories figure: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "ellipsoid_angle_trajectories_0_45_90.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'ellipsoid_angle_trajectories_0_45_90.png')}")
    print("=" * 95)

if __name__ == "__main__":
    main()

