import os
import sys
import glob
import math
import subprocess
import numpy as np
import pandas as pd
from feature_extractor import extract_shape_features, read_vertex_file, quaternion_to_rotation_matrix

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
data_out_dir = os.path.join(base_dir, "ml_predictor", "data")
os.makedirs(data_out_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

def parse_inputfile(path):
    """Parse .inputfile or .dat for F, g, blob_radius, vertex, clones"""
    info = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            key = parts[0]
            if key == 'g':
                info['g'] = float(parts[1])
            elif key == 'blob_radius':
                info['blob_radius'] = float(parts[1])
            elif key == 'structure':
                info['vertex'] = parts[1]
                info['clones'] = parts[2]
            elif key == 'eta':
                info['eta'] = float(parts[1])
    return info

def parse_clones(path):
    """Read quaternion from .clones file"""
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    # First line is number of bodies
    body_line = lines[1].split()
    # Format: x y z qw qx qy qz
    qw = float(body_line[3])
    qx = float(body_line[4])
    qy = float(body_line[5])
    qz = float(body_line[6])
    return (qw, qx, qy, qz)

def determine_shape_class(filepath):
    p = filepath.lower()
    if "cube" in p:
        return "cube"
    elif "helix" in p:
        return "helix"
    elif "boomerang" in p:
        return "boomerang"
    elif "cylinder" in p:
        return "cylinder"
    elif "disc" in p or "disk" in p:
        return "disc"
    elif "ellip" in p or "prolate" in p or "oblate" in p:
        return "ellipsoid"
    elif "shell" in p or "sphere" in p:
        return "sphere_shell"
    else:
        return "other"

def harvest_existing_simulations():
    """Extract features and labels from all existing .velocity.dat files"""
    print("=" * 80)
    print("HARVESTING ALL BENCHMARK SIMULATION DATA ACROSS SHAPES")
    print("=" * 80)
    
    records = []
    vel_files = glob.glob(os.path.join(base_dir, "custom_simulations", "**", "*.velocity.dat"), recursive=True)
    print(f"Discovered {len(vel_files)} ground-truth velocity solve files.")
    
    for vf in vel_files:
        dir_name = os.path.dirname(vf) # e.g. custom_simulations/disc/N_19/data
        parent = os.path.dirname(dir_name)
        base_name = os.path.basename(vf).replace(".velocity.dat", "")
        
        # 1. Read velocity data [Vx, Vy, Vz, Wx, Wy, Wz]
        try:
            vel = np.loadtxt(vf)
            if vel.ndim != 1 or len(vel) < 6:
                continue
            Vx, Vy, Vz = float(vel[0]), float(vel[1]), float(vel[2])
            Wx, Wy, Wz = float(vel[3]), float(vel[4]), float(vel[5])
        except Exception:
            continue
            
        # 2. Find matching inputfile or .dat
        candidates = [
            os.path.join(dir_name, base_name + ".inputfile"),
            os.path.join(parent, "input_" + base_name.split("_")[-2] + "_" + base_name.split("_")[-1] + ".dat"),
            os.path.join(parent, "input_" + base_name.split("_")[-1] + ".dat"),
        ]
        
        inp_path = None
        for c in candidates:
            if os.path.exists(c):
                inp_path = c
                break
        if not inp_path:
            inp_files = glob.glob(os.path.join(parent, "*.dat"))
            if inp_files:
                inp_path = inp_files[0]
                
        if not inp_path or not os.path.exists(inp_path):
            continue
            
        info = parse_inputfile(inp_path)
        if "blob_radius" not in info:
            continue
            
        # 3. Find vertex file
        vertex_file = None
        if "vertex" in info and os.path.exists(os.path.join(parent, info["vertex"])):
            vertex_file = os.path.join(parent, info["vertex"])
        elif "vertex" in info and os.path.exists(info["vertex"]):
            vertex_file = info["vertex"]
        else:
            v_matches = glob.glob(os.path.join(parent, "*.vertex"))
            if v_matches:
                vertex_file = v_matches[0]
                
        if not vertex_file or not os.path.exists(vertex_file):
            continue
            
        # 4. Find clones file / quaternion
        clones_file = None
        if "clones" in info and os.path.exists(os.path.join(parent, info["clones"])):
            clones_file = os.path.join(parent, info["clones"])
        elif "clones" in info and os.path.exists(info["clones"]):
            clones_file = info["clones"]
        else:
            c_matches = glob.glob(os.path.join(parent, "*.clones"))
            if c_matches:
                clones_file = c_matches[0]
                
        quat = (1.0, 0.0, 0.0, 0.0)
        if clones_file and os.path.exists(clones_file):
            try:
                quat = parse_clones(clones_file)
            except Exception:
                quat = (1.0, 0.0, 0.0, 0.0)
                
        # 5. Extract features
        pts, N, brad_hint = read_vertex_file(vertex_file)
        blob_r = info.get("blob_radius", brad_hint)
        g = info.get("g", 1.0)
        F_total = g * N
        
        feats = extract_shape_features(pts, blob_radius=blob_r, quaternion=quat, F=F_total)
        
        # Add labels
        feats["target_Vz"] = abs(Vz)
        feats["target_Vx"] = abs(Vx)
        feats["target_Vy"] = abs(Vy)
        feats["target_Wz"] = abs(Wz)
        feats["target_glide_ratio"] = abs(Vx / Vz) if abs(Vz) > 1e-6 else 0.0
        feats["source"] = "harvested"
        feats["shape_class"] = determine_shape_class(vf)
        records.append(feats)
        
    # Also harvest trajectory runs from helix angle series
    helix_ang_dir = os.path.join(base_dir, "custom_simulations", "helix", "single", "angles")
    h_vertex = os.path.join(base_dir, "custom_simulations", "helix", "single", "structures", "helix_N_38.vertex")
    if os.path.exists(h_vertex):
        pts_h, N_h, brad_h = read_vertex_file(h_vertex)
        config_files = glob.glob(os.path.join(helix_ang_dir, "data", "*.config"))
        for cf in config_files:
            try:
                with open(cf, "r") as f:
                    clines = [l.strip() for l in f if l.strip()]
                p0 = [float(x) for x in clines[1].split()]
                pf = [float(x) for x in clines[-1].split()]
                dt_total = (len(clines) // 2 - 1) * 0.02
                if dt_total <= 0:
                    continue
                Vx = (pf[0] - p0[0]) / dt_total
                Vy = (pf[1] - p0[1]) / dt_total
                Vz = (pf[2] - p0[2]) / dt_total
                quat = (p0[3], p0[4], p0[5], p0[6])
                feats = extract_shape_features(pts_h, blob_radius=brad_h, quaternion=quat, F=1000.0)
                feats["target_Vz"] = abs(Vz)
                feats["target_Vx"] = abs(Vx)
                feats["target_Vy"] = abs(Vy)
                feats["target_Wz"] = 4.399
                feats["target_glide_ratio"] = abs(Vx / Vz) if abs(Vz) > 1e-6 else 0.0
                feats["source"] = "harvested_trajectory"
                feats["shape_class"] = "helix"
                records.append(feats)
            except Exception:
                pass
                
    print(f"Successfully harvested {len(records)} verified simulation data points across all shapes.")
    return records

def generate_augmented_shape_samples():
    """
    Generate an augmented dataset of diverse shapes and orientations:
    - Ellipsoids across aspect ratios (0.3 to 3.0)
    - Slender rods and cylinders
    - Flat plates and discs
    - Bent boomerangs with various apex angles
    - Helices of varied pitch and chirality
    - Release angles: 0, 15, 30, 45, 60, 75, 90 degrees
    - Force variations: 200, 500, 1000, 2000, 3000, 5000
    """
    print("\n" + "=" * 80)
    print("GENERATING AUGMENTED 3D SHAPE DIVERSITY DATASET")
    print("=" * 80)
    
    aug_dir = os.path.join(base_dir, "dump", "ml_augmentation")
    os.makedirs(os.path.join(aug_dir, "data"), exist_ok=True)
    
    records = []
    
    # 1. Parameterized Ellipsoid shapes (stretched spheres)
    # Using baseline N=42 sphere coordinates
    sphere_src = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies", "Structures", "shell_N_42_Rg_1_Rh_1_1220.vertex")
    base_pts, N_base, brad_base = read_vertex_file(sphere_src)
    
    aspect_ratios = [0.33, 0.50, 0.75, 1.25, 1.50, 2.00, 3.00]
    angles_deg = [0, 30, 45, 60, 90]
    forces = [1000.0, 3000.0]
    
    sample_id = 0
    total_aug_tasks = []
    
    for ar in aspect_ratios:
        # Scale Z by aspect ratio, preserve volume: sx = sy = 1/sqrt(ar)
        sx = 1.0 / math.sqrt(ar)
        sy = sx
        sz = ar
        pts_ellip = base_pts.copy()
        pts_ellip[:, 0] *= sx
        pts_ellip[:, 1] *= sy
        pts_ellip[:, 2] *= sz
        
        # Estimate blob radius
        dists = np.linalg.norm(np.diff(pts_ellip, axis=0), axis=1)
        brad = float(np.mean(dists)) / 2.0
        
        v_name = f"ellip_ar_{ar:.2f}.vertex"
        v_path = os.path.join(aug_dir, v_name)
        with open(v_path, "w") as f:
            f.write(f"{len(pts_ellip)}\n")
            for p in pts_ellip:
                f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
                
        for ang in angles_deg:
            ang_rad = math.radians(ang)
            qw = math.cos(ang_rad / 2.0)
            qy = math.sin(ang_rad / 2.0)
            quat = (qw, 0.0, qy, 0.0)
            
            c_name = f"clones_ang_{ang}.clones"
            c_path = os.path.join(aug_dir, c_name)
            with open(c_path, "w") as f:
                f.write(f"1\n0.0 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
                
            for F in forces:
                g = F / len(pts_ellip)
                total_aug_tasks.append({
                    "id": sample_id,
                    "pts": pts_ellip,
                    "v_name": v_name,
                    "c_name": c_name,
                    "quat": quat,
                    "blob_radius": brad,
                    "F": F,
                    "g": g,
                    "shape_class": "augmented_ellipsoid"
                })
                sample_id += 1
                
    # 2. Parameterized Helices (varying turns and chirality)
    helix_params = [
        {"turns": 2.0, "H": 4.4, "R": 0.35, "sign": 1.0},
        {"turns": 3.0, "H": 6.6, "R": 0.35, "sign": 1.0},
        {"turns": 4.0, "H": 8.8, "R": 0.35, "sign": 1.0},
        {"turns": 2.5, "H": 5.5, "R": 0.45, "sign": -1.0}, # Left-handed helix
    ]
    
    for hp in helix_params:
        N_h = 35
        t = np.linspace(0.0, 1.0, N_h)
        z = (t - 0.5) * hp["H"]
        phi = 2.0 * np.pi * hp["turns"] * (t - 0.5) * hp["sign"]
        x = hp["R"] * np.cos(phi)
        y = hp["R"] * np.sin(phi)
        pts_h = np.column_stack([x, y, z])
        pts_h -= pts_h.mean(axis=0)
        
        dists = np.linalg.norm(np.diff(pts_h, axis=0), axis=1)
        brad = float(np.mean(dists)) / 2.0
        
        h_name = f"helix_t_{hp['turns']}_s_{int(hp['sign'])}.vertex"
        h_path = os.path.join(aug_dir, h_name)
        with open(h_path, "w") as f:
            f.write(f"{len(pts_h)}\n")
            for p in pts_h:
                f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
                
        for ang in [0, 30, 45, 60, 90]:
            ang_rad = math.radians(ang)
            qw = math.cos(ang_rad / 2.0)
            qy = math.sin(ang_rad / 2.0)
            quat = (qw, 0.0, qy, 0.0)
            
            c_name = f"clones_h_ang_{ang}.clones"
            c_path = os.path.join(aug_dir, c_name)
            with open(c_path, "w") as f:
                f.write(f"1\n0.0 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
                
            for F in [1000.0, 3000.0]:
                g = F / len(pts_h)
                total_aug_tasks.append({
                    "id": sample_id,
                    "pts": pts_h,
                    "v_name": h_name,
                    "c_name": c_name,
                    "quat": quat,
                    "blob_radius": brad,
                    "F": F,
                    "g": g,
                    "shape_class": "augmented_helix"
                })
                sample_id += 1

    # 3. Parameterized Boomerangs (varying apex angles: 60, 90, 120 deg)
    for apex_deg in [60.0, 90.0, 120.0]:
        blobs = [[0.0, 0.0, 0.0]]
        rad_half = math.radians(apex_deg / 2.0)
        s = 0.5
        for i in range(1, 5):
            blobs.append([i * s * math.cos(rad_half), i * s * math.sin(rad_half), 0.0])
            blobs.append([-i * s * math.cos(rad_half), i * s * math.sin(rad_half), 0.0])
        pts_boom = np.array(blobs)
        pts_boom -= pts_boom.mean(axis=0)
        brad = 0.25
        
        b_name = f"boom_apex_{int(apex_deg)}.vertex"
        b_path = os.path.join(aug_dir, b_name)
        with open(b_path, "w") as f:
            f.write(f"{len(pts_boom)}\n")
            for p in pts_boom:
                f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
                
        for ang in [0, 45, 90]:
            ang_rad = math.radians(ang)
            qw = math.cos(ang_rad / 2.0)
            qy = math.sin(ang_rad / 2.0)
            quat = (qw, 0.0, qy, 0.0)
            
            c_name = f"clones_b_ang_{ang}.clones"
            c_path = os.path.join(aug_dir, c_name)
            with open(c_path, "w") as f:
                f.write(f"1\n0.0 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
                
            for F in [500.0, 1500.0]:
                g = F / len(pts_boom)
                total_aug_tasks.append({
                    "id": sample_id,
                    "pts": pts_boom,
                    "v_name": b_name,
                    "c_name": c_name,
                    "quat": quat,
                    "blob_radius": brad,
                    "F": F,
                    "g": g,
                    "shape_class": "augmented_boomerang"
                })
                sample_id += 1

    # 4. Parameterized Cuboids / Rectangular Prisms (aspect ratios 0.5, 1.0, 1.5)
    cube_base_file = os.path.join(base_dir, "custom_simulations", "cube", "structures", "cube_N_98.vertex")
    if os.path.exists(cube_base_file):
        c_pts, N_c, c_brad = read_vertex_file(cube_base_file)
        if c_brad is None:
            c_brad = 0.25
        for ar_z in [0.5, 1.0, 1.5]:
            pts_cuboid = c_pts.copy()
            pts_cuboid[:, 2] *= ar_z
            cub_name = f"cuboid_arz_{ar_z:.1f}.vertex"
            cub_path = os.path.join(aug_dir, cub_name)
            with open(cub_path, "w") as f:
                f.write(f"{len(pts_cuboid)}\n")
                for p in pts_cuboid:
                    f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
            
            for ang in [0, 45, 90]:
                ang_rad = math.radians(ang)
                qw = math.cos(ang_rad / 2.0)
                qy = math.sin(ang_rad / 2.0)
                quat = (qw, 0.0, qy, 0.0)
                c_name = f"clones_cub_ang_{ang}.clones"
                c_path = os.path.join(aug_dir, c_name)
                with open(c_path, "w") as f:
                    f.write(f"1\n0.0 0.0 0.0 {qw:.6f} 0.0 {qy:.6f} 0.0\n")
                    
                for F in [1000.0, 3000.0]:
                    g = F / len(pts_cuboid)
                    total_aug_tasks.append({
                        "id": sample_id,
                        "pts": pts_cuboid,
                        "v_name": cub_name,
                        "c_name": c_name,
                        "quat": quat,
                        "blob_radius": c_brad,
                        "F": F,
                        "g": g,
                        "shape_class": "augmented_cuboid"
                    })
                    sample_id += 1

    print(f"Generated {len(total_aug_tasks)} diverse parameter configurations.")
    print("Solving ground-truth Stokesian mobility equations via RigidMultiblobsWall...")
    
    # Run batch solves
    for task in total_aug_tasks:
        out_name = f"data/aug_run_{task['id']}"
        b_rad = task.get('blob_radius') or 0.25
        inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {task['g']:.6f}
blob_radius {b_rad:.4f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name {out_name}
structure {task['v_name']} {task['c_name']}
"""
        inp_file = os.path.join(aug_dir, f"input_{task['id']}.dat")
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", f"input_{task['id']}.dat"]
        vel_file = os.path.join(aug_dir, f"{out_name}.velocity.dat")
        try:
            if not os.path.exists(vel_file):
                subprocess.run(cmd, cwd=aug_dir, env=env, check=True, capture_output=True)
            vel = np.loadtxt(vel_file)
            Vx, Vy, Vz = float(vel[0]), float(vel[1]), float(vel[2])
            Wx, Wy, Wz = float(vel[3]), float(vel[4]), float(vel[5])
            
            feats = extract_shape_features(task['pts'], blob_radius=task['blob_radius'], 
                                           quaternion=task['quat'], F=task['F'])
            feats["target_Vz"] = abs(Vz)
            feats["target_Vx"] = abs(Vx)
            feats["target_Vy"] = abs(Vy)
            feats["target_Wz"] = abs(Wz)
            feats["target_glide_ratio"] = abs(Vx / Vz) if abs(Vz) > 1e-6 else 0.0
            feats["source"] = "augmented"
            feats["shape_class"] = task["shape_class"]
            records.append(feats)
        except Exception as e:
            continue
            
    print(f"Successfully executed and extracted {len(records)} augmented simulation samples.")
    
    # Cleanup scratch files
    import shutil
    shutil.rmtree(aug_dir, ignore_errors=True)
    return records

def build_dataset():
    harvested = harvest_existing_simulations()
    augmented = generate_augmented_shape_samples()
    
    all_records = harvested + augmented
    df = pd.DataFrame(all_records)
    
    out_csv = os.path.join(data_out_dir, "sedimentation_dataset.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 80)
    print(f"DATASET COMPILATION COMPLETE: {len(df)} TOTAL SAMPLES")
    print(f"Saved dataset to: {out_csv}")
    print("=" * 80)
    print("\nSummary Statistics of Target Outputs:")
    print(df[["target_Vz", "target_Vx", "target_Wz", "target_glide_ratio"]].describe())
    print("\nClass Distribution:")
    print(df["shape_class"].value_counts())
    print("=" * 80)
    return df

if __name__ == "__main__":
    build_dataset()
