import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
data_path = os.path.join(base_dir, "ml_predictor", "data", "sedimentation_dataset.csv")
models_dir = os.path.join(base_dir, "ml_predictor", "models")
plots_dir = os.path.join(base_dir, "ml_predictor", "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

FEATURE_COLS = [
    "N", "blob_radius", "Lx", "Ly", "Lz", "aspect_ratio_max", "aspect_ratio_z",
    "V_box", "Rg", "D_max", "A_blobs", "V_blobs",
    "I_l1", "I_l2", "I_l3", "asphericity", "acylindricity", "rel_anisotropy",
    "theta_deg", "cos_theta", "sin_2theta",
    "A_perp", "A_parallel", "area_ratio", "chiral_index",
    "F", "g_force_per_blob"
]

def train_and_evaluate():
    print("=" * 85)
    print("MACHINE LEARNING SEDIMENTATION PREDICTOR: TRAINING & MULTI-MODEL BENCHMARK")
    print("=" * 85)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run dataset_builder.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} samples, {len(FEATURE_COLS)} input features.")
    
    # 1. Data Integrity & Missing Values Check
    null_counts = df[FEATURE_COLS].isnull().sum().sum()
    print(f"Total missing / NULL values across features: {null_counts}")
    if null_counts > 0:
        df = df.dropna(subset=FEATURE_COLS)
        
    X = df[FEATURE_COLS].values
    y_vz = df["target_Vz"].values
    y_vx = df["target_Vx"].values
    y_wz = df["target_Wz"].values
    
    # 2. Strict Train/Test Split (80% Train, 20% Holdout Test) BEFORE Scaling
    X_train, X_test, y_vz_train, y_vz_test, idx_train, idx_test = train_test_split(
        X, y_vz, np.arange(len(X)), test_size=0.20, random_state=42
    )
    y_vx_train, y_vx_test = y_vx[idx_train], y_vx[idx_test]
    y_wz_train, y_wz_test = y_wz[idx_train], y_wz[idx_test]
    
    print(f"Train samples: {len(X_train)}, Holdout Test samples: {len(X_test)}")
    
    # 3. Fit Preprocessing Scaler on Training Data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    
    # 4. Multi-Model Evaluation on Primary Target Vz
    model_candidates = {
        "Ridge Regression (Baseline)": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2500, early_stopping=True, random_state=42)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = {}
    
    print("\n" + "-" * 85)
    print(f"{'Model':<30} {'CV R2 (5-Fold)':<18} {'Test R2':<12} {'Test RMSE':<12} {'Test MAE':<10}")
    print("-" * 85)
    
    best_model_vz = None
    best_r2_vz = -1e9
    best_name_vz = ""
    
    test_preds_vz = {}
    
    for name, model in model_candidates.items():
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_vz_train, cv=cv, scoring="r2")
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        
        # Fit on full training set
        model.fit(X_train_scaled, y_vz_train)
        y_pred = model.predict(X_test_scaled)
        test_preds_vz[name] = y_pred
        
        r2 = r2_score(y_vz_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_vz_test, y_pred))
        mae = mean_absolute_error(y_vz_test, y_pred)
        
        benchmark_results[name] = {
            "cv_r2_mean": float(mean_cv),
            "cv_r2_std": float(std_cv),
            "test_r2": float(r2),
            "test_rmse": float(rmse),
            "test_mae": float(mae)
        }
        
        print(f"{name:<30} {mean_cv:.4f} +/- {std_cv:.4f}   {r2:<12.4f} {rmse:<12.3f} {mae:<10.3f}")
        
        if r2 > best_r2_vz:
            best_r2_vz = r2
            best_model_vz = model
            best_name_vz = name
            
    print("-" * 85)
    print(f"Optimal Vz Model: {best_name_vz} (Test R2 = {best_r2_vz:.4f})")
    
    # Save optimal Vz model
    joblib.dump(best_model_vz, os.path.join(models_dir, "model_Vz.joblib"))
    
    # 5. Train Specialized Models for Lateral Drift (Vx) and Chiral Spin (Wz)
    print("\nTraining Specialized Predictors for Lateral Drift (Vx) and Chiral Spin (Wz)...")
    
    model_vx = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    model_vx.fit(X_train_scaled, y_vx_train)
    y_vx_pred = model_vx.predict(X_test_scaled)
    r2_vx = r2_score(y_vx_test, y_vx_pred)
    rmse_vx = math.sqrt(mean_squared_error(y_vx_test, y_vx_pred))
    print(f"Lateral Drift Vx Model (XGBoost): Test R2 = {r2_vx:.4f}, RMSE = {rmse_vx:.4f}")
    joblib.dump(model_vx, os.path.join(models_dir, "model_Vx.joblib"))
    
    model_wz = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    model_wz.fit(X_train_scaled, y_wz_train)
    y_wz_pred = model_wz.predict(X_test_scaled)
    r2_wz = r2_score(y_wz_test, y_wz_pred)
    rmse_wz = math.sqrt(mean_squared_error(y_wz_test, y_wz_pred))
    print(f"Chiral Spin Wz Model (XGBoost):    Test R2 = {r2_wz:.4f}, RMSE = {rmse_wz:.4f}")
    joblib.dump(model_wz, os.path.join(models_dir, "model_Wz.joblib"))
    
    # 6. Save Metadata & Manifest
    metadata = {
        "feature_names": FEATURE_COLS,
        "n_samples": len(df),
        "n_features": len(FEATURE_COLS),
        "vz_model": best_name_vz,
        "metrics_vz": benchmark_results,
        "metrics_vx": {"test_r2": float(r2_vx), "test_rmse": float(rmse_vx)},
        "metrics_wz": {"test_r2": float(r2_wz), "test_rmse": float(rmse_wz)}
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    # 7. Generate Comprehensive 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: Parity Plot for Best Model Vz
    ax_a = axes[0, 0]
    ax_a.scatter(y_vz_test, test_preds_vz[best_name_vz], color="#1f77b4", edgecolors="k", alpha=0.75, s=60, label=f"Test Samples (N={len(y_vz_test)})")
    min_v = min(y_vz_test.min(), test_preds_vz[best_name_vz].min())
    max_v = max(y_vz_test.max(), test_preds_vz[best_name_vz].max())
    ax_a.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2.0, label="Exact 1:1 Agreement")
    ax_a.set_title(f"Parity Plot: Predicted vs Actual $V_z$ ({best_name_vz})", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Ground Truth Stokesian $V_z$ (units/s)", fontsize=11)
    ax_a.set_ylabel("ML Predicted $V_z$ (units/s)", fontsize=11)
    ax_a.grid(True, linestyle="--", alpha=0.6)
    ax_a.legend(fontsize=10)
    
    # Panel B: Residual Distribution
    ax_b = axes[0, 1]
    residuals = test_preds_vz[best_name_vz] - y_vz_test
    ax_b.hist(residuals, bins=15, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax_b.axvline(0, color="r", linestyle="--", linewidth=2.0)
    ax_b.set_title(f"Residual Distribution: Error = $\hat{{V}}_z - V_z$", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Prediction Residual (units/s)", fontsize=11)
    ax_b.set_ylabel("Frequency Count", fontsize=11)
    ax_b.grid(True, linestyle="--", alpha=0.6)
    
    # Panel C: Feature Importances (Tree-based)
    ax_c = axes[1, 0]
    rf_model = model_candidates["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_feats = [FEATURE_COLS[i] for i in indices]
    top_vals = importances[indices]
    
    y_pos = np.arange(len(top_feats))
    ax_c.barh(y_pos, top_vals, align="center", color="#ff7f0e", edgecolor="black", alpha=0.8)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(top_feats, fontsize=10)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Normalized Feature Importance (Gini)", fontsize=11)
    ax_c.set_title("Top 10 Physical Descriptors Driving Sedimentation Speed", fontsize=12, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.6)
    
    # Panel D: Lateral Drift (Vx) Parity
    ax_d = axes[1, 1]
    ax_d.scatter(y_vx_test, y_vx_pred, color="#9467bd", edgecolors="k", alpha=0.75, s=60, label=f"Test Samples (R2={r2_vx:.3f})")
    min_vx = min(y_vx_test.min(), y_vx_pred.min())
    max_vx = max(y_vx_test.max(), y_vx_pred.max())
    ax_d.plot([min_vx, max_vx], [min_vx, max_vx], "r--", linewidth=2.0, label="Exact 1:1 Line")
    ax_d.set_title("Parity Plot: Lateral Drift Velocity $V_x$", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Ground Truth Lateral Drift $V_x$ (units/s)", fontsize=11)
    ax_d.set_ylabel("ML Predicted Lateral Drift $V_x$ (units/s)", fontsize=11)
    ax_d.grid(True, linestyle="--", alpha=0.6)
    ax_d.legend(fontsize=10)
    
    plt.tight_layout()
    
    p1 = os.path.join(plots_dir, "model_comparison_and_parity.png")
    p2 = os.path.join(thesis_fig_dir, "ml_sedimentation_parity.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nGenerated evaluation plots:")
    print(f"  -> {p1}")
    print(f"  -> {p2}")
    print("=" * 85)

if __name__ == "__main__":
    train_and_evaluate()
