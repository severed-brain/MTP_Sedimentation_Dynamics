import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
struct_dir = os.path.join(mb_custom, "structures")
data_dir = os.path.join(mb_custom, "data")
plots_dir = os.path.join(mb_custom, "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

sys.path.insert(0, os.path.join(base_dir, "ml_predictor"))
from predict_multibody import MultiBodySedimentationPredictor

def run_cross_check(shape_name="hexagonal_prism"):
    print("=" * 95)
    print("MULTI-BODY AI/ML SURROGATE BENCHMARK & NUMERICAL CROSS-CHECK")
    print(f"Random Test Shape: {shape_name.replace('_', ' ').title()}")
    print("=" * 95)
    
    predictor = MultiBodySedimentationPredictor()
    
    # 1. Load Ground Truth Multi-Body Simulation Data
    # Supports shapes with .dat scaling files or trajectories_summary.json
    sim_data = []
    v0_sim = None
    
    dat_path = os.path.join(data_dir, shape_name, f"{shape_name}_pair_scaling.dat")
    if os.path.exists(dat_path):
        with open(dat_path, "r") as f:
            for line in f:
                if line.startswith("# Baseline Solitary Velocity:"):
                    v0_sim = float(line.split("=")[-1].replace("units/s", "").strip())
                elif not line.startswith("#") and line.strip():
                    p = line.split()
                    sim_data.append({
                        "d_r": float(p[0]),
                        "vz_sim": float(p[3]),
                        "ratio_sim": float(p[4]),
                        "stokeslet_th": float(p[5]),
                        "wallclock_s": float(p[7])
                    })
    elif shape_name == "helix":
        with open(os.path.join(data_dir, "helix", "helix_pair_summary.json")) as f:
            hdata = json.load(f)
        v0_sim = float(hdata["v0_single"])
        for dr, rat, vz in zip(hdata["d_ratios"], hdata["enhancements"], hdata["vz_pairs"]):
            sim_data.append({
                "d_r": float(dr),
                "vz_sim": float(vz),
                "ratio_sim": float(rat),
                "stokeslet_th": 1.0 + 0.75 / float(dr),
                "wallclock_s": 5.5
            })
            
    df_sim = pd.DataFrame(sim_data)
    
    # 2. Run ML Predictions across all separations d/R
    vertex_file = os.path.join(struct_dir, f"{shape_name}_N_74.vertex" if shape_name == "hexagonal_prism" else f"{shape_name}_N_64.vertex" if shape_name == "trefoil_knot" else f"{shape_name}_N_56.vertex")
    if not os.path.exists(vertex_file):
        for f in os.listdir(struct_dir):
            if f.startswith(shape_name) and f.endswith(".vertex"):
                vertex_file = os.path.join(struct_dir, f)
                break
                
    print(f"Vertex File: {os.path.basename(vertex_file)}")
    print(f"Solitary Ground Truth Velocity V0: {v0_sim:.4f} units/s")
    print("-" * 95)
    print(f"{'d/R':<7} {'Sim Vz (u/s)':<15} {'ML Vz (u/s)':<15} {'Vz Err (%)':<12} {'Sim Ratio':<12} {'ML Ratio':<12} {'Ratio Err (%)':<14} {'Speedup':<10}")
    print("-" * 95)
    
    results = []
    for _, row in df_sim.iterrows():
        dr = row["d_r"]
        preds = predictor.predict_pair(vertex_file, d_r=dr, F=1000.0, known_v0=v0_sim)
        
        sim_vz = row["vz_sim"]
        ml_vz = preds["V_pair"]
        sim_rat = row["ratio_sim"]
        ml_rat = preds["enhancement_ratio"]
        
        vz_err = abs(ml_vz - sim_vz) / sim_vz * 100.0
        rat_err = abs(ml_rat - sim_rat) / sim_rat * 100.0
        t_inf = max(preds["inference_time_ms"], 0.2)
        speedup = (row["wallclock_s"] * 1000.0) / t_inf
        
        results.append({
            "d_r": dr,
            "sim_vz": sim_vz,
            "ml_vz": ml_vz,
            "vz_err_pct": vz_err,
            "sim_ratio": sim_rat,
            "ml_ratio": ml_rat,
            "ratio_err_pct": rat_err,
            "stokeslet_th": row["stokeslet_th"],
            "wallclock_s": row["wallclock_s"],
            "infer_ms": preds["inference_time_ms"],
            "speedup": speedup
        })
        print(f"{dr:<7.1f} {sim_vz:<15.4f} {ml_vz:<15.4f} {vz_err:<12.2f}% {sim_rat:<12.4f} {ml_rat:<12.4f} {rat_err:<14.2f}% {speedup:<9.0f}x")
        
    df_res = pd.DataFrame(results)
    
    # Save CSV
    csv_out = os.path.join(mb_custom, "data", f"multibody_ml_vs_sim_{shape_name}.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved cross-check CSV: {csv_out}")
    
    # -------------------------------------------------------------------------
    # 3. Generate 4-Panel Comparison Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    d_dense = np.linspace(1.8, 10.5, 200)
    stokeslet_dense = 1.0 + 0.75 / d_dense
    
    # Panel A: Velocity Enhancement Ratio vs d/R
    ax_a = axes[0, 0]
    ax_a.plot(df_res["d_r"], df_res["sim_ratio"], 'ro-', linewidth=2.4, markersize=8, label="Numerical Simulation (RMBW)")
    ax_a.plot(df_res["d_r"], df_res["ml_ratio"], 'bs--', linewidth=2.0, markersize=7, label="ML Surrogate (Gradient Boosting)")
    ax_a.plot(d_dense, stokeslet_dense, 'k:', linewidth=2.0, label=r"Continuum Stokeslet ($1 + \frac{3}{4}\frac{R}{d}$)")
    ax_a.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label="Solitary Baseline ($V_0$)")
    ax_a.set_title(f"(A) Velocity Enhancement $V_z / V_0$ vs Separation $d/R$", fontsize=11, fontweight='bold')
    ax_a.set_xlabel("Separation Ratio $d/R$", fontsize=10)
    ax_a.set_ylabel("Hydrodynamic Ratio $V_z / V_0$", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(fontsize=9)
    
    # Panel B: Dimensional Terminal Velocity Vz vs d/R
    ax_b = axes[0, 1]
    ax_b.plot(df_res["d_r"], df_res["sim_vz"], 'ro-', linewidth=2.4, markersize=8, label="Numerical Simulation ($V_{z,\\text{pair}}$)")
    ax_b.plot(df_res["d_r"], df_res["ml_vz"], 'bs--', linewidth=2.0, markersize=7, label="ML Surrogate Prediction ($V_{z,\\text{pair}}$)")
    ax_b.axhline(v0_sim, color='purple', linestyle=':', linewidth=2.0, label=f"Solitary $V_0 = {v0_sim:.2f}$ u/s")
    ax_b.set_title(f"(B) Dimensional Settling Speed $V_z$ vs $d/R$", fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Separation Ratio $d/R$", fontsize=10)
    ax_b.set_ylabel("Settling Velocity $V_z$ (units/s)", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    ax_b.legend(fontsize=9)
    
    # Panel C: Parity Plot (Simulation vs ML)
    ax_c = axes[1, 0]
    ax_c.scatter(df_res["sim_vz"], df_res["ml_vz"], color="#2ca02c", s=100, edgecolors='k', zorder=3, label=f"{shape_name.capitalize()} Pairs")
    min_v = min(df_res["sim_vz"].min(), df_res["ml_vz"].min()) * 0.95
    max_v = max(df_res["sim_vz"].max(), df_res["ml_vz"].max()) * 1.05
    ax_c.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2.0, label="1:1 Exact Parity")
    
    r2_val = 1.0 - np.sum((df_res["ml_vz"] - df_res["sim_vz"])**2) / np.sum((df_res["sim_vz"] - np.mean(df_res["sim_vz"]))**2)
    ax_c.set_title(f"(C) Multi-Body Parity: Simulation vs ML ($R^2 = {r2_val:.4f}$)", fontsize=11, fontweight='bold')
    ax_c.set_xlabel("Ground Truth Simulation $V_z$ (units/s)", fontsize=10)
    ax_c.set_ylabel("ML Surrogate Predicted $V_z$ (units/s)", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.6)
    ax_c.legend(fontsize=9)
    
    # Panel D: Relative Error & Computational Speedup
    ax_d = axes[1, 1]
    ax_d.plot(df_res["d_r"], df_res["vz_err_pct"], 'g^-', linewidth=2.2, markersize=8, label="Velocity Relative Error (%)")
    ax_d.set_title(f"(D) Prediction Accuracy & Computational Speedup", fontsize=11, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $d/R$", fontsize=10)
    ax_d.set_ylabel("Relative Error (%)", fontsize=10, color='g')
    ax_d.tick_params(axis='y', labelcolor='g')
    ax_d.set_ylim([0, 10])
    ax_d.grid(True, linestyle="--", alpha=0.6)
    
    ax_d2 = ax_d.twinx()
    ax_d2.plot(df_res["d_r"], df_res["speedup"], 'm*--', linewidth=2.0, markersize=8, label="Speedup Factor")
    ax_d2.set_ylabel("Acceleration Factor (x times faster)", fontsize=10, color='m')
    ax_d2.tick_params(axis='y', labelcolor='m')
    
    lines_1, labels_1 = ax_d.get_legend_handles_labels()
    lines_2, labels_2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    p1 = os.path.join(plots_dir, f"multibody_ml_vs_sim_{shape_name}.png")
    p2 = os.path.join(thesis_fig_dir, f"multibody_ml_vs_sim_{shape_name}.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated comparison figures:\n  -> {p1}\n  -> {p2}")
    print("-" * 95)
    print("CROSS-CHECK SUMMARY METRICS:")
    print(f"  * Average Velocity Relative Error: {df_res['vz_err_pct'].mean():.2f}%")
    print(f"  * Maximum Velocity Relative Error: {df_res['vz_err_pct'].max():.2f}%")
    print(f"  * Parity Correlation R2:           {r2_val:.4f}")
    print(f"  * Average Speedup Factor:          {df_res['speedup'].mean():,.0f}x faster!")
    print("=" * 95)

if __name__ == "__main__":
    shape_to_test = sys.argv[1] if len(sys.argv) > 1 else "hexagonal_prism"
    run_cross_check(shape_to_test)
