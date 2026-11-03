import os
import sys
import math
import json
import numpy as np
import pandas as pd

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_data = os.path.join(base_dir, "custom_simulations", "multi_body", "data")
struct_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
data_out_dir = os.path.join(base_dir, "ml_predictor", "data")
os.makedirs(data_out_dir, exist_ok=True)

sys.path.insert(0, os.path.join(base_dir, "ml_predictor"))
from feature_extractor import extract_shape_features, read_vertex_file

# Configuration for all 9 benchmark shapes
shape_configs = {
    "sphere": {
        "vertex": "sphere_N_42.vertex",
        "blob_radius": 0.2735,
        "F": 1000.0,
        "R": 1.0
    },
    "ellipsoid": {
        "vertex": "ellipsoid_prolate_N_42.vertex",
        "blob_radius": 0.2735,
        "F": 1000.0,
        "R": 1.0
    },
    "disc": {
        "vertex": "disc_N_37.vertex",
        "blob_radius": 0.1650,
        "F": 1000.0,
        "R": 1.0
    },
    "cylinder": {
        "vertex": "cylinder_N_86.vertex",
        "blob_radius": 0.0742,
        "F": 1000.0,
        "R": 1.0
    },
    "boomerang": {
        "vertex": "boomerang_N_15.vertex",
        "blob_radius": 0.3245,
        "F": 300.0,
        "R": 1.0
    },
    "helix": {
        "vertex": "helix_N_38.vertex",
        "blob_radius": 0.1254,
        "F": 1000.0,
        "R": 1.0
    },
    "cube": {
        "vertex": "cube_N_56.vertex",
        "blob_radius": 0.3333,
        "F": 1000.0,
        "R": 1.0
    },
    "hexagonal_prism": {
        "vertex": "hexagonal_prism_N_74.vertex",
        "blob_radius": 0.1601,
        "F": 1000.0,
        "R": 1.0
    },
    "trefoil_knot": {
        "vertex": "trefoil_knot_N_64.vertex",
        "blob_radius": 0.1288,
        "F": 1000.0,
        "R": 1.0
    }
}

def get_vz_from_traj(traj, times):
    z_start = traj[0][2]
    z_end = traj[-1][2]
    return abs((z_end - z_start) / (times[-1] - times[0]))

def build_multibody_dataset():
    print("=" * 85)
    print("HARVESTING MULTI-BODY SIMULATION DATA ACROSS ALL 9 SHAPES")
    print("=" * 85)
    
    # Pre-load vertex points
    shape_pts = {}
    for name, cfg in shape_configs.items():
        vpath = os.path.join(struct_dir, cfg["vertex"])
        pts, N, _ = read_vertex_file(vpath)
        shape_pts[name] = pts
        print(f"  Loaded {name:<16}: N = {N:<4} blobs from {cfg['vertex']}")
        
    records = []
    
    # -------------------------------------------------------------------------
    # 1. Trajectories summary JSON (Sphere, Cylinder, Disc, Ellipsoid, Boomerang)
    # -------------------------------------------------------------------------
    summary_path = os.path.join(mb_data, "trajectories_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            ts_data = json.load(f)
            
        # Spheres, Cylinders, Discs
        for s in ["sphere", "cylinder", "disc"]:
            cfg = shape_configs[s]
            pts = shape_pts[s]
            feats = extract_shape_features(pts, blob_radius=cfg["blob_radius"], F=cfg["F"])
            s_sim = ts_data[s]["single"]["0.0"]
            v0 = get_vz_from_traj(s_sim["traj_b1"], s_sim["times"])
            
            for d_str, sim in ts_data[s]["pair"].items():
                d_r = float(d_str)
                t = sim["times"]
                vz_b1 = get_vz_from_traj(sim["traj_b1"], t)
                vz_b2 = get_vz_from_traj(sim["traj_b2"], t)
                vz_mean = (vz_b1 + vz_b2) / 2.0
                
                rec = dict(feats)
                rec["shape"] = s
                rec["d_r"] = d_r
                rec["inv_d_r"] = 1.0 / d_r
                rec["inv_d_r3"] = (1.0 / d_r) ** 3
                rec["rel_yaw_deg"] = 0.0
                rec["cos_rel_yaw"] = 1.0
                rec["single_Vz"] = v0
                rec["target_pair_Vz"] = vz_mean
                rec["target_ratio"] = vz_mean / v0
                rec["target_delta_Vz"] = vz_mean - v0
                records.append(rec)
                
        # Ellipsoid (orientations: 0, 45, 90 deg)
        for ang_str in ["ang_0", "ang_45", "ang_90"]:
            ang_val = float(ang_str.replace("ang_", ""))
            cfg = shape_configs["ellipsoid"]
            pts = shape_pts["ellipsoid"]
            ang_rad = math.radians(ang_val)
            quat = (math.cos(ang_rad / 2.0), 0.0, math.sin(ang_rad / 2.0), 0.0)
            feats = extract_shape_features(pts, blob_radius=cfg["blob_radius"], quaternion=quat, F=cfg["F"])
            
            s_sim = ts_data["ellipsoid"][ang_str]["0.0"]
            v0 = get_vz_from_traj(s_sim["traj_b1"], s_sim["times"])
            
            for d_str in ["2.0", "4.0", "6.0", "8.0", "10.0"]:
                if d_str in ts_data["ellipsoid"][ang_str]:
                    sim = ts_data["ellipsoid"][ang_str][d_str]
                    t = sim["times"]
                    vz_b1 = get_vz_from_traj(sim["traj_b1"], t)
                    vz_b2 = get_vz_from_traj(sim["traj_b2"], t)
                    vz_mean = (vz_b1 + vz_b2) / 2.0
                    
                    rec = dict(feats)
                    rec["shape"] = "ellipsoid"
                    rec["d_r"] = float(d_str)
                    rec["inv_d_r"] = 1.0 / float(d_str)
                    rec["inv_d_r3"] = (1.0 / float(d_str)) ** 3
                    rec["rel_yaw_deg"] = 0.0
                    rec["cos_rel_yaw"] = 1.0
                    rec["single_Vz"] = v0
                    rec["target_pair_Vz"] = vz_mean
                    rec["target_ratio"] = vz_mean / v0
                    rec["target_delta_Vz"] = vz_mean - v0
                    records.append(rec)
                    
        # Boomerang (relative orientations: 0, 45, 90, 180 deg)
        cfg = shape_configs["boomerang"]
        pts = shape_pts["boomerang"]
        feats = extract_shape_features(pts, blob_radius=cfg["blob_radius"], F=cfg["F"])
        s_sim = ts_data["boomerang"]["single"]["0.0"]
        v0 = get_vz_from_traj(s_sim["traj_b1"], s_sim["times"])
        
        for rel_str in ["rel_0", "rel_45", "rel_90", "rel_180"]:
            rel_val = float(rel_str.replace("rel_", ""))
            for d_str in ["2.0", "4.0", "6.0", "8.0", "10.0"]:
                if d_str in ts_data["boomerang"][rel_str]:
                    sim = ts_data["boomerang"][rel_str][d_str]
                    t = sim["times"]
                    vz_b1 = get_vz_from_traj(sim["traj_b1"], t)
                    vz_b2 = get_vz_from_traj(sim["traj_b2"], t)
                    vz_mean = (vz_b1 + vz_b2) / 2.0
                    
                    rec = dict(feats)
                    rec["shape"] = "boomerang"
                    rec["d_r"] = float(d_str)
                    rec["inv_d_r"] = 1.0 / float(d_str)
                    rec["inv_d_r3"] = (1.0 / float(d_str)) ** 3
                    rec["rel_yaw_deg"] = rel_val
                    rec["cos_rel_yaw"] = math.cos(math.radians(rel_val))
                    rec["single_Vz"] = v0
                    rec["target_pair_Vz"] = vz_mean
                    rec["target_ratio"] = vz_mean / v0
                    rec["target_delta_Vz"] = vz_mean - v0
                    records.append(rec)

    # -------------------------------------------------------------------------
    # 2. Helix
    # -------------------------------------------------------------------------
    helix_json = os.path.join(mb_data, "helix", "helix_pair_summary.json")
    if os.path.exists(helix_json):
        cfg = shape_configs["helix"]
        pts = shape_pts["helix"]
        feats = extract_shape_features(pts, blob_radius=cfg["blob_radius"], F=cfg["F"])
        with open(helix_json, "r") as f:
            hdata = json.load(f)
        v0 = float(hdata["v0_single"])
        for dr, ratio, vz in zip(hdata["d_ratios"], hdata["enhancements"], hdata["vz_pairs"]):
            rec = dict(feats)
            rec["shape"] = "helix"
            rec["d_r"] = float(dr)
            rec["inv_d_r"] = 1.0 / float(dr)
            rec["inv_d_r3"] = (1.0 / float(dr)) ** 3
            rec["rel_yaw_deg"] = 0.0
            rec["cos_rel_yaw"] = 1.0
            rec["single_Vz"] = v0
            rec["target_pair_Vz"] = float(vz)
            rec["target_ratio"] = float(ratio)
            rec["target_delta_Vz"] = float(vz) - v0
            records.append(rec)

    # -------------------------------------------------------------------------
    # 3. Cube, Hexagonal Prism, Trefoil Knot
    # -------------------------------------------------------------------------
    for s in ["cube", "hexagonal_prism", "trefoil_knot"]:
        cfg = shape_configs[s]
        pts = shape_pts[s]
        feats = extract_shape_features(pts, blob_radius=cfg["blob_radius"], F=cfg["F"])
        dat_path = os.path.join(mb_data, s, f"{s}_pair_scaling.dat")
        if os.path.exists(dat_path):
            with open(dat_path, "r") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    dr = float(parts[0])
                    vz = float(parts[3])
                    ratio = float(parts[4])
                    v0 = vz / ratio
                    
                    rec = dict(feats)
                    rec["shape"] = s
                    rec["d_r"] = dr
                    rec["inv_d_r"] = 1.0 / dr
                    rec["inv_d_r3"] = (1.0 / dr) ** 3
                    rec["rel_yaw_deg"] = 0.0
                    rec["cos_rel_yaw"] = 1.0
                    rec["single_Vz"] = v0
                    rec["target_pair_Vz"] = vz
                    rec["target_ratio"] = ratio
                    rec["target_delta_Vz"] = vz - v0
                    records.append(rec)

    df = pd.DataFrame(records)
    csv_out = os.path.join(data_out_dir, "multibody_pair_dataset.csv")
    df.to_csv(csv_out, index=False)
    
    print("-" * 85)
    print(f"SUCCESS: Harvested {len(df)} multi-body pair simulation points across {df['shape'].nunique()} shape classes.")
    print(f"Dataset saved to: {csv_out}")
    print("-" * 85)
    print("Breakdown by Shape:")
    for s, count in df["shape"].value_counts().items():
        print(f"  * {s:<18}: {count} pair simulations")
    print("=" * 85)
    return df

if __name__ == "__main__":
    build_multibody_dataset()
