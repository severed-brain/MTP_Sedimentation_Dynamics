import os
import sys
import math
import time
import json
import argparse
import subprocess
import joblib
import numpy as np

# Ensure local module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_shape_features, read_vertex_file

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
models_dir = os.path.join(base_dir, "ml_predictor", "models")

class SedimentationPredictor:
    def __init__(self, models_dir=models_dir):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.model_vz = joblib.load(os.path.join(models_dir, "model_Vz.joblib"))
        self.model_vx = joblib.load(os.path.join(models_dir, "model_Vx.joblib"))
        self.model_wz = joblib.load(os.path.join(models_dir, "model_Wz.joblib"))
        
        with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]

    def predict_from_features(self, feat_dict):
        """Run ML inference given a feature dictionary"""
        x_vec = np.array([[feat_dict[k] for k in self.feature_names]], dtype=np.float64)
        x_scaled = self.scaler.transform(x_vec)
        
        t0 = time.time()
        pred_vz = float(self.model_vz.predict(x_scaled)[0])
        pred_vx = float(self.model_vx.predict(x_scaled)[0])
        pred_wz = float(self.model_wz.predict(x_scaled)[0])
        t_infer = (time.time() - t0) * 1000.0 # ms
        
        # Physical constraints
        pred_vz = max(pred_vz, 0.0)
        pred_vx = max(pred_vx, 0.0)
        pred_wz = max(pred_wz, 0.0)
        glide_ratio = pred_vx / pred_vz if pred_vz > 1e-6 else 0.0
        
        return {
            "Vz": pred_vz,
            "Vx": pred_vx,
            "Wz": pred_wz,
            "glide_ratio": glide_ratio,
            "inference_time_ms": t_infer
        }

    def predict_shape(self, vertex_path, blob_radius=None, angle_deg=0.0, axis='y', F=1000.0):
        """
        Predict sedimentation velocity, lateral drift, and chiral spin for an arbitrary .vertex file.
        """
        pts, N, brad_hint = read_vertex_file(vertex_path)
        if blob_radius is None:
            blob_radius = brad_hint
            
        # Orientation quaternion
        ang_rad = math.radians(angle_deg)
        qw = math.cos(ang_rad / 2.0)
        s = math.sin(ang_rad / 2.0)
        if axis == 'y':
            quat = (qw, 0.0, s, 0.0)
        elif axis == 'x':
            quat = (qw, s, 0.0, 0.0)
        else:
            quat = (qw, 0.0, 0.0, s)
            
        feats = extract_shape_features(pts, blob_radius=blob_radius, quaternion=quat, F=F)
        preds = self.predict_from_features(feats)
        return feats, preds

def validate_against_simulation(vertex_path, blob_radius, angle_deg, F):
    """Run actual RigidMultiblobsWall solve to measure ground-truth error"""
    scratch_dir = os.path.join(base_dir, "dump", "ml_validation_scratch")
    os.makedirs(os.path.join(scratch_dir, "data"), exist_ok=True)
    
    mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"
    
    import shutil
    v_dst = os.path.join(scratch_dir, os.path.basename(vertex_path))
    shutil.copyfile(vertex_path, v_dst)
    
    ang_rad = math.radians(angle_deg)
    qw = math.cos(ang_rad / 2.0)
    qy = math.sin(ang_rad / 2.0)
    
    with open(os.path.join(scratch_dir, "val.clones"), "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
        
    pts, N, brad = read_vertex_file(vertex_path)
    g = F / N
    
    inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {g:.6f}
blob_radius {blob_radius:.4f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name data/val_sim
structure {os.path.basename(vertex_path)} val.clones
"""
    with open(os.path.join(scratch_dir, "input.dat"), "w") as f:
        f.write(inp_content)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", "input.dat"]
    t0 = time.time()
    subprocess.run(cmd, cwd=scratch_dir, env=env, check=True, capture_output=True)
    sim_time = time.time() - t0
    
    vel = np.loadtxt(os.path.join(scratch_dir, "data", "val_sim.velocity.dat"))
    shutil.rmtree(scratch_dir, ignore_errors=True)
    
    return {
        "Vz": abs(float(vel[2])),
        "Vx": abs(float(vel[0])),
        "Wz": abs(float(vel[5])),
        "glide_ratio": abs(float(vel[0]) / float(vel[2])) if abs(float(vel[2])) > 1e-6 else 0.0,
        "sim_time_s": sim_time
    }

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Predictor for Arbitrary Rigid Body Sedimentation")
    parser.add_argument("--vertex", type=str, required=True, help="Path to arbitrary .vertex file")
    parser.add_argument("--blob-radius", type=float, default=None, help="Hydrodynamic blob radius (optional)")
    parser.add_argument("--angle", type=float, default=0.0, help="Inclination pitch angle in degrees (default: 0)")
    parser.add_argument("--force", type=float, default=1000.0, help="Gravitational force magnitude (default: 1000)")
    parser.add_argument("--validate", action="store_true", help="Run numerical simulation solve to validate accuracy")
    args = parser.parse_args()
    
    if not os.path.exists(args.vertex):
        print(f"Error: Vertex file '{args.vertex}' not found.")
        sys.exit(1)
        
    predictor = SedimentationPredictor()
    feats, preds = predictor.predict_shape(args.vertex, blob_radius=args.blob_radius, 
                                           angle_deg=args.angle, F=args.force)
    
    print("=" * 80)
    print("AI SEDIMENTATION DYNAMICS PREDICTOR (ARBITRARY SINGLE SHAPE)")
    print("=" * 80)
    print(f"Input Shape:      {os.path.basename(args.vertex)}")
    print(f"Discretization:   N = {feats['N']:.0f} blobs, a = {feats['blob_radius']:.4f}")
    print(f"Dimensions:       Lx = {feats['Lx']:.2f}, Ly = {feats['Ly']:.2f}, Lz = {feats['Lz']:.2f}")
    print(f"Shape Invariants: Aspect Ratio = {feats['aspect_ratio_max']:.2f}, Anisotropy = {feats['rel_anisotropy']:.3f}, Chirality = {feats['chiral_index']:.4f}")
    print(f"Orientation:      Pitch Angle theta = {args.angle:.1f} deg (cos theta = {feats['cos_theta']:.3f})")
    print(f"Applied Force:    F = {args.force:.1f} units")
    print("-" * 80)
    print("MACHINE LEARNING PREDICTION RESULTS (Inference time: {:.2f} ms):".format(preds['inference_time_ms']))
    print(f"  * Vertical Terminal Velocity Vz:  {preds['Vz']:8.3f} units/s")
    print(f"  * Lateral Drift Velocity Vx:       {preds['Vx']:8.3f} units/s")
    print(f"  * Chiral Angular Velocity Wz:      {preds['Wz']:8.3f} rad/s")
    print(f"  * Glide Ratio |Vx / Vz|:           {preds['glide_ratio']:8.4f}")
    
    if args.validate:
        print("\n" + "-" * 80)
        print("RUNNING GROUND-TRUTH STOKESIAN SIMULATION FOR VALIDATION...")
        sim_res = validate_against_simulation(args.vertex, feats['blob_radius'], args.angle, args.force)
        
        vz_err = abs(preds['Vz'] - sim_res['Vz']) / sim_res['Vz'] * 100.0 if sim_res['Vz'] > 1e-6 else 0.0
        vx_err = abs(preds['Vx'] - sim_res['Vx'])
        
        print(f"Simulation completed in {sim_res['sim_time_s']:.2f} s. Comparison Table:")
        print(f"{'Quantity':<30} {'ML Predicted':<15} {'Ground Truth':<15} {'Rel Error':<12}")
        print("-" * 75)
        print(f"{'Vertical Velocity Vz (units/s)':<30} {preds['Vz']:<15.3f} {sim_res['Vz']:<15.3f} {vz_err:.2f}%")
        print(f"{'Lateral Drift Vx (units/s)':<30} {preds['Vx']:<15.3f} {sim_res['Vx']:<15.3f} {vx_err:.3f} abs")
        print(f"{'Chiral Spin Wz (rad/s)':<30} {preds['Wz']:<15.3f} {sim_res['Wz']:<15.3f}")
        print(f"{'Glide Ratio |Vx/Vz|':<30} {preds['glide_ratio']:<15.4f} {sim_res['glide_ratio']:<15.4f}")
        speedup = (sim_res['sim_time_s'] * 1000.0) / max(preds['inference_time_ms'], 0.01)
        print(f"\n=> Computational Acceleration: {speedup:,.0f}x faster than numerical simulation!")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
