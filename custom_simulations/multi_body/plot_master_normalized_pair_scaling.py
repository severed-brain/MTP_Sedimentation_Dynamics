import os
import sys
import json
import csv
import math
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
    traj_data = json.load(f)

# Helper to compute velocity from trajectory
def get_vz(traj, times):
    z_start = traj[0][2]
    z_end = traj[-1][2]
    return abs((z_end - z_start) / (times[-1] - times[0]))

def main():
    print("=" * 95)
    print("MASTER NORMALIZED PAIR SEDIMENTATION SCALING (V_z / V_0) ACROSS ALL SHAPES")
    print("Explicitly Normalizing by Single Particle Baseline V_0 with Continuum Theory Comparison")
    print("=" * 95)
    
    d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    # Extract baseline V0 and pair velocities
    shapes_data = {}
    
    # 1. Sphere
    v0_sph = get_vz(traj_data["sphere"]["single"]["0.0"]["traj_b1"], traj_data["sphere"]["single"]["0.0"]["times"])
    vz_sph = [get_vz(traj_data["sphere"]["pair"][str(d)]["traj_b1"], traj_data["sphere"]["pair"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Sphere"] = {
        "V0": v0_sph, "Vz": vz_sph, "ratio": [v / v0_sph for v in vz_sph],
        "color": "#d62728", "marker": "o", "label": f"Sphere (V₀ = {v0_sph:.1f} u/s)"
    }
    
    # 2. Ellipsoid theta = 0
    v0_ell0 = get_vz(traj_data["ellipsoid"]["ang_0"]["0.0"]["traj_b1"], traj_data["ellipsoid"]["ang_0"]["0.0"]["times"])
    vz_ell0 = [get_vz(traj_data["ellipsoid"]["ang_0"][str(d)]["traj_b1"], traj_data["ellipsoid"]["ang_0"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Ellipsoid (θ=0°)"] = {
        "V0": v0_ell0, "Vz": vz_ell0, "ratio": [v / v0_ell0 for v in vz_ell0],
        "color": "#1f77b4", "marker": "s", "label": f"Ellipsoid θ=0° (V₀ = {v0_ell0:.1f} u/s)"
    }
    
    # 3. Ellipsoid theta = 45
    v0_ell45 = get_vz(traj_data["ellipsoid"]["ang_45"]["0.0"]["traj_b1"], traj_data["ellipsoid"]["ang_45"]["0.0"]["times"])
    vz_ell45 = [get_vz(traj_data["ellipsoid"]["ang_45"][str(d)]["traj_b1"], traj_data["ellipsoid"]["ang_45"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Ellipsoid (θ=45°)"] = {
        "V0": v0_ell45, "Vz": vz_ell45, "ratio": [v / v0_ell45 for v in vz_ell45],
        "color": "#ff7f0e", "marker": "^", "label": f"Ellipsoid θ=45° (V₀ = {v0_ell45:.1f} u/s)"
    }
    
    # 4. Ellipsoid theta = 90
    v0_ell90 = get_vz(traj_data["ellipsoid"]["ang_90"]["0.0"]["traj_b1"], traj_data["ellipsoid"]["ang_90"]["0.0"]["times"])
    vz_ell90 = [get_vz(traj_data["ellipsoid"]["ang_90"][str(d)]["traj_b1"], traj_data["ellipsoid"]["ang_90"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Ellipsoid (θ=90°)"] = {
        "V0": v0_ell90, "Vz": vz_ell90, "ratio": [v / v0_ell90 for v in vz_ell90],
        "color": "#2ca02c", "marker": "v", "label": f"Ellipsoid θ=90° (V₀ = {v0_ell90:.1f} u/s)"
    }
    
    # 5. Disc
    v0_disc = get_vz(traj_data["disc"]["single"]["0.0"]["traj_b1"], traj_data["disc"]["single"]["0.0"]["times"])
    vz_disc = [get_vz(traj_data["disc"]["pair"][str(d)]["traj_b1"], traj_data["disc"]["pair"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Disc"] = {
        "V0": v0_disc, "Vz": vz_disc, "ratio": [v / v0_disc for v in vz_disc],
        "color": "#9467bd", "marker": "p", "label": f"Disc Broadside (V₀ = {v0_disc:.1f} u/s)"
    }
    
    # 6. Cylinder
    v0_cyl = get_vz(traj_data["cylinder"]["single"]["0.0"]["traj_b1"], traj_data["cylinder"]["single"]["0.0"]["times"])
    vz_cyl = [get_vz(traj_data["cylinder"]["pair"][str(d)]["traj_b1"], traj_data["cylinder"]["pair"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Cylinder"] = {
        "V0": v0_cyl, "Vz": vz_cyl, "ratio": [v / v0_cyl for v in vz_cyl],
        "color": "#8c564b", "marker": "d", "label": f"Cylinder (V₀ = {v0_cyl:.1f} u/s)"
    }
    
    # 7. Cube
    cube_dat = os.path.join(data_dir, "cube", "cube_pair_scaling.dat")
    if os.path.exists(cube_dat):
        with open(cube_dat, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        v0_cube = 33.104417
        vz_cube = [float(l.split()[3]) for l in lines]
        shapes_data["Cube"] = {
            "V0": v0_cube, "Vz": vz_cube, "ratio": [v / v0_cube for v in vz_cube],
            "color": "#e377c2", "marker": "s", "label": f"Cube (V₀ = {v0_cube:.1f} u/s)"
        }
        
    # 8. Helix
    hlx_summary = os.path.join(data_dir, "helix", "helix_pair_summary.json")
    if os.path.exists(hlx_summary):
        with open(hlx_summary, "r") as f:
            hlx_info = json.load(f)
        v0_hlx = hlx_info["v0_single"]
        vz_hlx = hlx_info["vz_pairs"]
        shapes_data["Helix"] = {
            "V0": v0_hlx, "Vz": vz_hlx, "ratio": [v / v0_hlx for v in vz_hlx],
            "color": "#17becf", "marker": "h", "label": f"Helix (V₀ = {v0_hlx:.1f} u/s)"
        }
        
    # 9. Boomerang
    v0_boom = get_vz(traj_data["boomerang"]["single"]["0.0"]["traj_b1"], traj_data["boomerang"]["single"]["0.0"]["times"])
    vz_boom = [get_vz(traj_data["boomerang"]["rel_0"][str(d)]["traj_b1"], traj_data["boomerang"]["rel_0"][str(d)]["times"]) for d in d_ratios]
    shapes_data["Boomerang (Δφ=0°)"] = {
        "V0": v0_boom, "Vz": vz_boom, "ratio": [v / v0_boom for v in vz_boom],
        "color": "#bcbd22", "marker": "*", "label": f"Boomerang (V₀ = {v0_boom:.1f} u/s)"
    }
    
    # -------------------------------------------------------------------------
    # GENERATE 2-PANEL PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    
    # Panel A: Normalized Velocity Enhancement V_z / V_0
    ax_a = axes[0]
    for s_name, s in shapes_data.items():
        ax_a.plot(d_ratios, s["ratio"], marker=s["marker"], color=s["color"],
                  linewidth=2.0, markersize=8, label=s["label"])
                  
    d_dense = np.linspace(1.8, 10.5, 200)
    stokeslet = 1.0 + 0.75 * (1.0 / d_dense)
    stimson_jeffery = 1.0 + 0.75 * (1.0 / d_dense) + 0.5625 * ((1.0 / d_dense)**2)
    
    ax_a.plot(d_dense, stokeslet, 'k--', linewidth=2.0, label=r'Stokeslet Asymptotic ($1 + \frac{3}{4}\frac{R}{d}$)')
    ax_a.plot(d_dense, stimson_jeffery, 'k:', linewidth=1.8, label=r'Stimson-Jeffery 2nd-Order')
    ax_a.axhline(1.0, color='gray', linestyle='-', linewidth=1.2, alpha=0.8, label='Solitary Particle Baseline (1.0)')
    
    ax_a.set_title(r"(A) Normalized Drafting Factor $V_z / V_0$ vs $d/R$ Across Particulate Shapes", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax_a.set_ylabel(r"Normalized Sedimentation Velocity $V_z / V_0$", fontsize=11)
    ax_a.set_xticks(d_ratios)
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(fontsize=8, loc="upper right", ncol=2)
    
    # Panel B: Absolute Settling Velocity V_z vs d/R with Single-Particle Baseline Lines
    ax_b = axes[1]
    for s_name, s in shapes_data.items():
        ax_b.plot(d_ratios, s["Vz"], marker=s["marker"], color=s["color"],
                  linewidth=2.0, markersize=8, label=f"{s_name} Pair")
        ax_b.axhline(s["V0"], color=s["color"], linestyle=':', linewidth=1.5, alpha=0.7)
        
    ax_b.set_title(r"(B) Absolute Terminal Velocity $|V_z|$ vs $d/R$ with Baseline $V_0$ (Dotted)", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Separation Distance Ratio $d/R$", fontsize=11)
    ax_b.set_ylabel("Terminal Velocity $|V_z|$ (units/s)", fontsize=11)
    ax_b.set_xticks(d_ratios)
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(fontsize=8, loc="upper right", ncol=2)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "master_normalized_pair_scaling.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated master normalized pair scaling plot: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "master_normalized_pair_scaling.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'master_normalized_pair_scaling.png')}")
    print("=" * 95)

if __name__ == "__main__":
    main()
