import os
import sys
import math
import time
import json
import shutil
import argparse
import subprocess
import joblib
import numpy as np

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
models_dir = os.path.join(base_dir, "ml_predictor", "models")
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")

sys.path.insert(0, os.path.join(base_dir, "ml_predictor"))
from feature_extractor import extract_shape_features, read_vertex_file

class MultiBodySedimentationPredictor:
    def __init__(self, models_dir=models_dir):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler_multibody.joblib"))
        self.model_ratio = joblib.load(os.path.join(models_dir, "model_multibody_ratio.joblib"))
        
        # Load single-body predictor to automatically predict V0 if solitary velocity is unknown
        self.single_scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.single_model_vz = joblib.load(os.path.join(models_dir, "model_Vz.joblib"))
        
        with open(os.path.join(models_dir, "multibody_model_metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names_ratio"]
        
        with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
            self.single_metadata = json.load(f)
        self.single_feature_names = self.single_metadata["feature_names"]

    def predict_pair(self, vertex_path, d_r=2.0, blob_radius=None, angle_deg=0.0, rel_yaw_deg=0.0, F=1000.0, known_v0=None):
        """
        Predict multi-body pair sedimentation dynamics:
        Returns:
            - V0: solitary single-body terminal velocity
            - V_pair: multi-body terminal settling velocity at separation d/R
            - ratio: hydrodynamic drafting enhancement factor V_pair / V0
            - delta_V: velocity boost (V_pair - V0)
            - inference_time_ms: computation time in milliseconds
        """
        pts, N, brad_hint = read_vertex_file(vertex_path)
        if blob_radius is None:
            blob_radius = brad_hint
            
        ang_rad = math.radians(angle_deg)
        quat = (math.cos(ang_rad / 2.0), 0.0, math.sin(ang_rad / 2.0), 0.0)
        
        # 1. Extract physical descriptors
        feats = extract_shape_features(pts, blob_radius=blob_radius, quaternion=quat, F=F)
        
        # 2. Get Solitary Velocity V0
        t0 = time.time()
        if known_v0 is not None:
            v0 = float(known_v0)
        else:
            x_single = np.array([[feats[k] for k in self.single_feature_names]], dtype=np.float64)
            x_single_scaled = self.single_scaler.transform(x_single)
            v0 = float(self.single_model_vz.predict(x_single_scaled)[0])
            v0 = max(v0, 0.1)
            
        # 3. Formulate Multi-Body Pair Features
        feat_dict = dict(feats)
        feat_dict["d_r"] = float(d_r)
        feat_dict["inv_d_r"] = 1.0 / float(d_r)
        feat_dict["inv_d_r3"] = (1.0 / float(d_r)) ** 3
        feat_dict["rel_yaw_deg"] = float(rel_yaw_deg)
        feat_dict["cos_rel_yaw"] = math.cos(math.radians(rel_yaw_deg))
        
        x_vec = np.array([[feat_dict[k] for k in self.feature_names]], dtype=np.float64)
        x_scaled = self.scaler.transform(x_vec)
        
        pred_ratio = float(self.model_ratio.predict(x_scaled)[0])
        # Hydrodynamic lower bound: ratio must be >= 1.0 in creeping flow drafting
        pred_ratio = max(pred_ratio, 1.0)
        
        pred_vz_pair = pred_ratio * v0
        delta_v = pred_vz_pair - v0
        t_infer = (time.time() - t0) * 1000.0 # ms
        
        return {
            "d_r": d_r,
            "V0": v0,
            "V_pair": pred_vz_pair,
            "enhancement_ratio": pred_ratio,
            "delta_Vz": delta_v,
            "inference_time_ms": t_infer,
            "N": N,
            "blob_radius": blob_radius
        }

def validate_pair_against_simulation(vertex_path, d_r, blob_radius, F, angle_deg=0.0, dt=0.02, n_steps=50):
    """Run ground-truth numerical multi-body simulation for the pair"""
    scratch_dir = os.path.join(base_dir, "dump", "mb_val_scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"
    
    pts, N, _ = read_vertex_file(vertex_path)
    g_val = F / N
    v_dst = os.path.join(scratch_dir, os.path.basename(vertex_path))
    shutil.copyfile(vertex_path, v_dst)
    
    # 1. Run Single Body Baseline
    clones_single = os.path.join(scratch_dir, "single.clones")
    with open(clones_single, "w") as f:
        f.write("1\n0.0 0.0 0.0 1.0 0.0 0.0 0.0\n")
        
    inp_single = os.path.join(scratch_dir, "inp_single.dat")
    with open(inp_single, "w") as f:
        f.write(f"""scheme deterministic_forward_euler
domain no_wall
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
blob_blob_force_implementation None
dt {dt}
n_steps {n_steps}
n_save 1
eta 1.0
g {g_val:.14f}
blob_radius {blob_radius}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
save_clones one_file
output_name sim_single
structure {os.path.basename(v_dst)} single.clones
""")
        
    cmd_s = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", "inp_single.dat"]
    subprocess.run(cmd_s, cwd=scratch_dir, env=env, check=True, capture_output=True)
    
    # Read solitary trajectory
    cfg_s = os.path.join(scratch_dir, "sim_single.sim_single.config")
    with open(cfg_s, "r") as f:
        lines_s = [l.strip() for l in f if l.strip()]
    traj_s = []
    idx = 0
    while idx < len(lines_s):
        nb = int(lines_s[idx].split()[0])
        idx += 1
        traj_s.append([float(x) for x in lines_s[idx].split()])
        idx += nb
    v0_sim = abs((traj_s[-1][2] - traj_s[0][2]) / (dt * n_steps))
    
    # 2. Run Pair Simulation
    R_char = 1.0
    d = d_r * R_char
    x1 = -d / 2.0
    x2 = +d / 2.0
    
    ang_rad = math.radians(angle_deg)
    qw = math.cos(ang_rad / 2.0)
    qy = math.sin(ang_rad / 2.0)
    
    clones_pair = os.path.join(scratch_dir, "pair.clones")
    with open(clones_pair, "w") as f:
        f.write(f"2\n{x1:.6f} 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n{x2:.6f} 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
        
    inp_pair = os.path.join(scratch_dir, "inp_pair.dat")
    with open(inp_pair, "w") as f:
        f.write(f"""scheme deterministic_forward_euler
domain no_wall
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
blob_blob_force_implementation None
dt {dt}
n_steps {n_steps}
n_save 1
eta 1.0
g {g_val:.14f}
blob_radius {blob_radius}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
save_clones one_file
output_name sim_pair
structure {os.path.basename(v_dst)} pair.clones
""")
        
    cmd_p = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", "inp_pair.dat"]
    t0 = time.time()
    subprocess.run(cmd_p, cwd=scratch_dir, env=env, check=True, capture_output=True)
    sim_time = time.time() - t0
    
    # Read pair trajectory
    cfg_p = os.path.join(scratch_dir, "sim_pair.sim_pair.config")
    with open(cfg_p, "r") as f:
        lines_p = [l.strip() for l in f if l.strip()]
    tb1, tb2 = [], []
    idx = 0
    while idx < len(lines_p):
        nb = int(lines_p[idx].split()[0])
        idx += 1
        tb1.append([float(x) for x in lines_p[idx].split()])
        tb2.append([float(x) for x in lines_p[idx+1].split()])
        idx += nb
        
    vz1 = abs((tb1[-1][2] - tb1[0][2]) / (dt * n_steps))
    vz2 = abs((tb2[-1][2] - tb2[0][2]) / (dt * n_steps))
    vz_pair_sim = (vz1 + vz2) / 2.0
    ratio_sim = vz_pair_sim / v0_sim
    
    shutil.rmtree(scratch_dir, ignore_errors=True)
    
    return {
        "v0_sim": v0_sim,
        "vz_pair_sim": vz_pair_sim,
        "ratio_sim": ratio_sim,
        "sim_time_s": sim_time
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Body Sedimentation AI/ML Surrogate Predictor")
    parser.add_argument("--vertex", type=str, required=True, help="Path to .vertex file")
    parser.add_argument("--distance", type=float, default=2.0, help="Separation distance ratio d/R (default: 2.0)")
    parser.add_argument("--blob-radius", type=float, default=None, help="Hydrodynamic blob radius (optional)")
    parser.add_argument("--force", type=float, default=1000.0, help="Gravitational force (default: 1000)")
    parser.add_argument("--angle", type=float, default=0.0, help="Pitch angle (default: 0)")
    parser.add_argument("--rel-yaw", type=float, default=0.0, help="Relative yaw angle (default: 0)")
    parser.add_argument("--sweep", action="store_true", help="Run full distance sweep [2, 4, 6, 8, 10]")
    parser.add_argument("--validate", action="store_true", help="Run ground-truth numerical Stokesian simulation")
    args = parser.parse_args()
    
    if not os.path.exists(args.vertex):
        print(f"Error: {args.vertex} does not exist.")
        sys.exit(1)
        
    predictor = MultiBodySedimentationPredictor()
    d_list = [2.0, 4.0, 6.0, 8.0, 10.0] if args.sweep else [args.distance]
    
    print("=" * 85)
    print("AI MULTI-BODY SEDIMENTATION SURROGATE PREDICTOR (ARBITRARY 3D PARTICLES)")
    print("=" * 85)
    print(f"Input Shape:      {os.path.basename(args.vertex)}")
    print(f"Applied Force:    F = {args.force} units")
    print("-" * 85)
    
    for d_r in d_list:
        preds = predictor.predict_pair(args.vertex, d_r=d_r, blob_radius=args.blob_radius,
                                       angle_deg=args.angle, rel_yaw_deg=args.rel_yaw, F=args.force)
        
        print(f"\n[Separation d/R = {d_r:.1f}] ML PREDICTION (Inference: {preds['inference_time_ms']:.2f} ms):")
        print(f"  * Baseline Solitary Velocity V0:     {preds['V0']:8.4f} units/s")
        print(f"  * Multi-Body Pair Settling Vz:       {preds['V_pair']:8.4f} units/s")
        print(f"  * Hydrodynamic Enhancement Factor:   {preds['enhancement_ratio']:8.4f}x (+{preds['delta_Vz']:5.2f} u/s)")
        
        if args.validate:
            print("  Running Ground-Truth Stokesian Simulation...")
            sim = validate_pair_against_simulation(args.vertex, d_r=d_r, blob_radius=preds['blob_radius'],
                                                   F=args.force, angle_deg=args.angle)
            
            vz_err = abs(preds['V_pair'] - sim['vz_pair_sim']) / sim['vz_pair_sim'] * 100.0
            rat_err = abs(preds['enhancement_ratio'] - sim['ratio_sim']) / sim['ratio_sim'] * 100.0
            speedup = (sim['sim_time_s'] * 1000.0) / preds['inference_time_ms']
            
            print(f"  Simulation Completed in {sim['sim_time_s']:.2f} s. Numerical Comparison:")
            print(f"  {'Metric':<25} {'ML Predicted':<15} {'Ground Truth':<15} {'Rel Error':<12}")
            print(f"  {'-'*67}")
            print(f"  {'Pair Settling Vz (u/s)':<25} {preds['V_pair']:<15.4f} {sim['vz_pair_sim']:<15.4f} {vz_err:<11.2f}%")
            print(f"  {'Enhancement Factor V/V0':<25} {preds['enhancement_ratio']:<15.4f} {sim['ratio_sim']:<15.4f} {rat_err:<11.2f}%")
            print(f"  => Computational Speedup: {speedup:,.0f}x faster than numerical solver!")
            
    print("\n" + "=" * 85)

if __name__ == "__main__":
    main()

