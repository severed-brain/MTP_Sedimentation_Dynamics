import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
data_path = os.path.join(base_dir, "ml_predictor", "data", "multibody_pair_dataset.csv")
models_dir = os.path.join(base_dir, "ml_predictor", "models")
plots_dir = os.path.join(base_dir, "ml_predictor", "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

# 32 Features (27 single-body descriptors + 5 pair configuration descriptors)
feature_names = [
    "N", "blob_radius", "Lx", "Ly", "Lz", "aspect_ratio_max", "aspect_ratio_z",
    "V_box", "Rg", "D_max", "A_blobs", "V_blobs", "I_l1", "I_l2", "I_l3",
    "asphericity", "acylindricity", "rel_anisotropy", "theta_deg", "cos_theta",
    "sin_2theta", "A_perp", "A_parallel", "area_ratio", "chiral_index", "F",
    "g_force_per_blob", "d_r", "inv_d_r", "inv_d_r3", "rel_yaw_deg", "cos_rel_yaw"
]

def train_and_evaluate():
    print("=" * 85)
    print("TRAINING MULTI-BODY SEDIMENTATION ML SURROGATE MODELS")
    print("Dataset: 70 Verified Pair Stokesian Simulations across 9 3D Shape Classes")
    print("=" * 85)
    
    df = pd.read_csv(data_path)
    X = df[feature_names].values
    y_ratio = df["target_ratio"].values
    y_vz = df["target_pair_Vz"].values
    v0_vals = df["single_Vz"].values
    shapes = df["shape"].values
    
    # -------------------------------------------------------------------------
    # 1. 5-Fold Cross-Validation Benchmark
    # -------------------------------------------------------------------------
    print("\n[PART 1] 5-Fold Cross-Validation Multi-Model Benchmark (Enhancement Ratio)")
    print("-" * 85)
    print(f"{'Algorithm':<28} {'CV R2':<12} {'CV MAE':<12} {'CV RMSE':<12} {'Max Error':<12}")
    print("-" * 85)
    
    algorithms = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42),
        "Gradient Boosting (GBR)": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42)
    }
    
    cv_metrics = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in algorithms.items():
        r2_scores, maes, rmses, max_errs = [], [], [], []
        for train_idx, val_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])
            
            model.fit(X_tr, y_ratio[train_idx])
            preds = model.predict(X_va)
            
            r2_scores.append(r2_score(y_ratio[val_idx], preds))
            maes.append(mean_absolute_error(y_ratio[val_idx], preds))
            rmses.append(np.sqrt(mean_squared_error(y_ratio[val_idx], preds)))
            max_errs.append(np.max(np.abs(preds - y_ratio[val_idx])))
            
        mean_r2 = np.mean(r2_scores)
        mean_mae = np.mean(maes)
        mean_rmse = np.mean(rmses)
        mean_max = np.mean(max_errs)
        
        cv_metrics[name] = {
            "cv_r2": float(mean_r2),
            "cv_mae": float(mean_mae),
            "cv_rmse": float(mean_rmse)
        }
        print(f"{name:<28} {mean_r2:<12.4f} {mean_mae:<12.4f} {mean_rmse:<12.4f} {mean_max:<12.4f}")
        
    # -------------------------------------------------------------------------
    # 2. Leave-One-Shape-Out Cross-Validation (Out-of-Distribution Generalization)
    # -------------------------------------------------------------------------
    print("\n[PART 2] Leave-One-Shape-Out (LOSO) Generalization Benchmark")
    print("Testing whether training on 8 shapes predicts the 9th unseen shape:")
    print("-" * 85)
    print(f"{'Holdout Shape':<20} {'V0 (u/s)':<12} {'Ratio Rel Err':<16} {'Vz MAE (u/s)':<16} {'Vz Rel Err':<14}")
    print("-" * 85)
    
    loso_results = {}
    loso_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42)
    
    for s in np.unique(shapes):
        te_idx = np.where(shapes == s)[0]
        tr_idx = np.where(shapes != s)[0]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        
        loso_model.fit(X_tr, y_ratio[tr_idx])
        pred_rat = loso_model.predict(X_te)
        
        pred_vz = pred_rat * v0_vals[te_idx]
        true_vz = y_vz[te_idx]
        
        rat_err = np.mean(np.abs(pred_rat - y_ratio[te_idx]) / y_ratio[te_idx]) * 100.0
        vz_mae = mean_absolute_error(true_vz, pred_vz)
        vz_err = np.mean(np.abs(pred_vz - true_vz) / true_vz) * 100.0
        v0_val = np.mean(v0_vals[te_idx])
        
        loso_results[s] = {
            "v0": float(v0_val),
            "ratio_err_pct": float(rat_err),
            "vz_mae": float(vz_mae),
            "vz_err_pct": float(vz_err)
        }
        print(f"{s:<20} {v0_val:<12.2f} {rat_err:<16.2f}% {vz_mae:<16.3f} {vz_err:<14.2f}%")
        
    # -------------------------------------------------------------------------
    # 3. Fit Final Production Models on Full Multi-Body Dataset
    # -------------------------------------------------------------------------
    print("\n[PART 3] Fitting Final Production Multi-Body Models...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    # 1. Optimal Enhancement Ratio Predictor (GBR)
    model_ratio = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    model_ratio.fit(X_scaled, y_ratio)
    
    # 2. Optimal Direct Velocity Predictor (GBR)
    # Include solitary velocity as an explicit feature for direct Vz solve
    feat_names_vz = feature_names + ["single_Vz"]
    X_vz = df[feat_names_vz].values
    scaler_vz = StandardScaler()
    X_vz_scaled = scaler_vz.fit_transform(X_vz)
    
    model_vz = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    model_vz.fit(X_vz_scaled, y_vz)
    
    # Save model artifacts
    joblib.dump(scaler_final, os.path.join(models_dir, "scaler_multibody.joblib"))
    joblib.dump(scaler_vz, os.path.join(models_dir, "scaler_multibody_vz.joblib"))
    joblib.dump(model_ratio, os.path.join(models_dir, "model_multibody_ratio.joblib"))
    joblib.dump(model_vz, os.path.join(models_dir, "model_multibody_Vz.joblib"))
    
    # Metadata JSON
    metadata = {
        "feature_names_ratio": feature_names,
        "feature_names_vz": feat_names_vz,
        "n_samples": len(df),
        "n_shapes": int(df["shape"].nunique()),
        "cv_metrics": cv_metrics,
        "loso_metrics": loso_results
    }
    with open(os.path.join(models_dir, "multibody_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved all model artifacts to ml_predictor/models/")
    
    # -------------------------------------------------------------------------
    # 4. Generate Comprehensive 4-Panel Parity & Scaling Figure
    # -------------------------------------------------------------------------
    print("\nGenerating 4-panel multi-body parity and validation plot...")
    preds_ratio = model_ratio.predict(X_scaled)
    preds_vz = preds_ratio * v0_vals
    
    r2_final_rat = r2_score(y_ratio, preds_ratio)
    r2_final_vz = r2_score(y_vz, preds_vz)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel A: Parity for Multi-Body Settling Velocity Vz
    ax_a = axes[0, 0]
    shape_list = np.unique(shapes)
    colors = plt.cm.tab10(np.linspace(0, 1, len(shape_list)))
    
    for i, s in enumerate(shape_list):
        idx = np.where(shapes == s)[0]
        ax_a.scatter(y_vz[idx], preds_vz[idx], color=colors[i], s=75, edgecolors='k', alpha=0.85, label=s.capitalize())
        
    min_v = min(y_vz.min(), preds_vz.min()) * 0.9
    max_v = max(y_vz.max(), preds_vz.max()) * 1.1
    ax_a.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2.0, label="1:1 Parity")
    ax_a.set_title(f"(A) Multi-Body Pair Sedimentation Parity ($R^2 = {r2_final_vz:.4f}$)", fontsize=11, fontweight='bold')
    ax_a.set_xlabel("Ground Truth Stokesian Pair $V_z$ (units/s)", fontsize=10)
    ax_a.set_ylabel("ML Surrogate Predicted Pair $V_z$ (units/s)", fontsize=10)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(fontsize=8, loc='upper left')
    
    # Panel B: Parity for Hydrodynamic Enhancement Ratio V_pair / V0
    ax_b = axes[0, 1]
    for i, s in enumerate(shape_list):
        idx = np.where(shapes == s)[0]
        ax_b.scatter(y_ratio[idx], preds_ratio[idx], color=colors[i], s=75, edgecolors='k', alpha=0.85, label=s.capitalize())
    
    min_r = min(y_ratio.min(), preds_ratio.min()) * 0.95
    max_r = max(y_ratio.max(), preds_ratio.max()) * 1.05
    ax_b.plot([min_r, max_r], [min_r, max_r], 'r--', linewidth=2.0, label="1:1 Parity")
    ax_b.set_title(f"(B) Hydrodynamic Enhancement Factor Parity ($R^2 = {r2_final_rat:.4f}$)", fontsize=11, fontweight='bold')
    ax_b.set_xlabel("Ground Truth Enhancement Ratio $V_z / V_0$", fontsize=10)
    ax_b.set_ylabel("ML Surrogate Predicted Ratio $V_z / V_0$", fontsize=10)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    ax_b.legend(fontsize=8, loc='upper left')
    
    # Panel C: Hydrodynamic Drafting Scaling: Simulation vs ML
    ax_c = axes[1, 0]
    test_shapes = ["sphere", "cube", "hexagonal_prism", "trefoil_knot", "octagonal_prism"]
    markers = {"sphere": "o", "cube": "s", "hexagonal_prism": "^", "trefoil_knot": "d", "octagonal_prism": "p"}
    
    for s in test_shapes:
        idx = np.where((shapes == s) & (df["rel_yaw_deg"] == 0.0))[0]
        sub = df.iloc[idx].sort_values("d_r")
        sub_preds = preds_ratio[idx][np.argsort(df.iloc[idx]["d_r"])]
        
        ax_c.plot(sub["d_r"], sub["target_ratio"], marker=markers[s], linewidth=2.0, label=f"Sim: {s.capitalize()}")
        ax_c.plot(sub["d_r"], sub_preds, '--', marker=markers[s], alpha=0.75, linewidth=1.5, label=f"ML: {s.capitalize()}")
        
    ax_c.set_title("(C) Multi-Body Drafting vs Distance $d/R$: Simulation vs ML", fontsize=11, fontweight='bold')
    ax_c.set_xlabel("Separation Ratio $d/R$", fontsize=10)
    ax_c.set_ylabel("Enhancement Factor $V_z / V_0$", fontsize=10)
    ax_c.grid(True, linestyle="--", alpha=0.6)
    ax_c.legend(fontsize=8, loc='upper right')
    
    # Panel D: Relative Error across Separation Distances
    ax_d = axes[1, 1]
    df["vz_rel_err"] = np.abs(preds_vz - y_vz) / y_vz * 100.0
    err_by_d = df.groupby("d_r")["vz_rel_err"].mean()
    err_max_d = df.groupby("d_r")["vz_rel_err"].max()
    
    ax_d.plot(err_by_d.index, err_by_d.values, 'ro-', linewidth=2.2, markersize=8, label="Mean Relative Error (%)")
    ax_d.plot(err_max_d.index, err_max_d.values, 'bs--', linewidth=1.8, markersize=7, label="Max Relative Error (%)")
    ax_d.set_title("(D) Prediction Error Distribution vs Separation Distance", fontsize=11, fontweight='bold')
    ax_d.set_xlabel("Separation Ratio $d/R$", fontsize=10)
    ax_d.set_ylabel("Relative Error (%)", fontsize=10)
    ax_d.set_ylim([0, 15])
    ax_d.grid(True, linestyle="--", alpha=0.6)
    ax_d.legend(fontsize=9)
    
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "multibody_model_parity.png")
    p2 = os.path.join(thesis_fig_dir, "multibody_model_parity.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight')
    plt.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated multi-body parity figures:\n  -> {p1}\n  -> {p2}")
    print("=" * 85)

if __name__ == "__main__":
    train_and_evaluate()

