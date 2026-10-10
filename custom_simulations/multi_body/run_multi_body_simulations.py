import os
import sys
import math
import time
import json
import argparse
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

base_dir = r"d:\sedimentation_dynamics"
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
mb_custom = os.path.join(base_dir, "custom_simulations", "multi_body")
struct_dir = os.path.join(mb_custom, "structures")
data_dir = os.path.join(mb_custom, "data")
os.makedirs(data_dir, exist_ok=True)

# Distance ratios d/R
d_ratios = [2.0, 4.0, 6.0, 8.0, 10.0]

# Geometries configurations
# name: (vertex_file, N, blob_radius, F_total, characteristic_R)
geom_configs = {
    "sphere": {
        "vertex": "sphere_N_42.vertex",
        "N": 42,
        "blob_radius": 0.2735,
        "F": 1000.0,
        "R": 1.0,
        "angles": [0] # symmetric
    },
    "ellipsoid": {
        "vertex": "ellipsoid_prolate_N_42.vertex",
        "N": 42,
        "blob_radius": 0.2735,
        "F": 1000.0,
        "R": 1.0,
        "angles": [0, 45, 90] # pitch tilt angles in degrees
    },
    "disc": {
        "vertex": "disc_N_37.vertex",
        "N": 37,
        "blob_radius": 0.165,
        "F": 1000.0,
        "R": 1.0,
        "angles": [0] # broadside horizontal
    },
    "cylinder": {
        "vertex": "cylinder_N_86.vertex",
        "N": 86,
        "blob_radius": 0.0742,
        "F": 1000.0,
        "R": 1.0, # semi-length
        "angles": [0] # along x-axis
    },
    "boomerang": {
        "vertex": "boomerang_N_15.vertex",
        "N": 15,
        "blob_radius": 0.3245,
        "F": 300.0,
        "R": 1.0,
        "angles": [0, 45, 90, 180] # relative yaw angles in degrees
    }
}

def make_quaternion_y(angle_deg):
    """Quaternion for pitch angle around Y-axis (tilts Z towards X)"""
    rad = math.radians(angle_deg)
    qw = math.cos(rad / 2.0)
    qx = 0.0
    qy = math.sin(rad / 2.0)
    qz = 0.0
    return (qw, qx, qy, qz)

def make_quaternion_z(angle_deg):
    """Quaternion for yaw angle around Z-axis (relative rotation in XY plane)"""
    rad = math.radians(angle_deg)
    qw = math.cos(rad / 2.0)
    qx = 0.0
    qy = 0.0
    qz = math.sin(rad / 2.0)
    return (qw, qx, qy, qz)

def run_single_simulation(task):
    geom, tag, d_r, q1, q2, is_single, dt, n_steps, rerun = task
    cfg_info = geom_configs[geom]
    v_file = os.path.join(struct_dir, cfg_info["vertex"])
    N = cfg_info["N"]
    R = cfg_info["R"]
    F = cfg_info["F"]
    g = F / N
    brad = cfg_info["blob_radius"]
    
    sim_dir = os.path.join(data_dir, geom)
    os.makedirs(sim_dir, exist_ok=True)
    
    out_base = f"{geom}_{tag}_d_{int(d_r)}" if not is_single else f"{geom}_{tag}_single"
    clones_file = os.path.join(sim_dir, f"{out_base}.clones")
    inp_file = os.path.join(sim_dir, f"input_{out_base}.dat")
    config_file = os.path.join(sim_dir, f"{out_base}.{out_base}.config")
    
    if not rerun and os.path.exists(config_file):
        pass # already done
    else:
        # Write clones file
        if is_single:
            clones_content = f"1\n0.0 0.0 0.0 {q1[0]:.6f} {q1[1]:.6f} {q1[2]:.6f} {q1[3]:.6f}\n"
        else:
            d = d_r * R
            x1 = -d / 2.0
            x2 = +d / 2.0
            clones_content = f"2\n{x1:.6f} 0.0 0.0 {q1[0]:.6f} {q1[1]:.6f} {q1[2]:.6f} {q1[3]:.6f}\n{x2:.6f} 0.0 0.0 {q2[0]:.6f} {q2[1]:.6f} {q2[2]:.6f} {q2[3]:.6f}\n"
            
        with open(clones_file, "w") as f:
            f.write(clones_content)
            
        inp_content = f"""scheme                                   deterministic_forward_euler
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       {dt}
n_steps                                  {n_steps}
n_save                                   1
eta                                      1.0
g                                        {g}
blob_radius                              {brad}
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              {out_base}
structure {v_file} {clones_file}
"""
        with open(inp_file, "w") as f:
            f.write(inp_content)
            
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"
        cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies.py"), "--input-file", f"input_{out_base}.dat"]
        subprocess.run(cmd, cwd=sim_dir, env=env, check=True, capture_output=True)

    # Parse trajectory
    with open(config_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    idx = 0
    traj_b1 = [] # [x, y, z, qw, qx, qy, qz]
    traj_b2 = []
    times = []
    step = 0
    while idx < len(lines):
        nb = int(lines[idx].split()[0])
        idx += 1
        bodies = []
        for _ in range(nb):
            bodies.append([float(x) for x in lines[idx].split()])
            idx += 1
        traj_b1.append(bodies[0])
        if nb > 1:
            traj_b2.append(bodies[1])
        times.append(step * dt)
        step += 1
        
    return {
        "geom": geom,
        "tag": tag,
        "d_r": d_r,
        "is_single": is_single,
        "times": times,
        "traj_b1": traj_b1,
        "traj_b2": traj_b2
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-body sedimentation simulation runner")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step dt")
    parser.add_argument("--steps", type=int, default=50, help="Number of integration steps")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--rerun", action="store_true", help="Force rerun simulations")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTI-BODY SEDIMENTATION TRAJECTORY SUITE (FORWARD EULER)")
    print(f"Time step: dt={args.dt}, Steps: {args.steps}, Duration: T={args.dt*args.steps:.2f} s")
    print(f"Distance ratios d/R: {d_ratios}")
    print("=" * 80)
    
    tasks = []
    
    # 1. Sphere: symmetric, angle=0
    for d_r in d_ratios:
        q = (1.0, 0.0, 0.0, 0.0)
        tasks.append(("sphere", "pair", d_r, q, q, False, args.dt, args.steps, args.rerun))
    tasks.append(("sphere", "single", 0.0, q, q, True, args.dt, args.steps, args.rerun))
    
    # 2. Ellipsoid: pitch angles 0, 45, 90 deg
    for ang in [0, 45, 90]:
        q = make_quaternion_y(ang)
        for d_r in d_ratios:
            tasks.append(("ellipsoid", f"ang_{ang}", d_r, q, q, False, args.dt, args.steps, args.rerun))
        tasks.append(("ellipsoid", f"ang_{ang}", 0.0, q, q, True, args.dt, args.steps, args.rerun))
        
    # 3. Disc: broadside horizontal (angle=0)
    q = (1.0, 0.0, 0.0, 0.0)
    for d_r in d_ratios:
        tasks.append(("disc", "pair", d_r, q, q, False, args.dt, args.steps, args.rerun))
    tasks.append(("disc", "single", 0.0, q, q, True, args.dt, args.steps, args.rerun))
    
    # 4. Cylinder: along x-axis (angle=0)
    q = (1.0, 0.0, 0.0, 0.0)
    for d_r in d_ratios:
        tasks.append(("cylinder", "pair", d_r, q, q, False, args.dt, args.steps, args.rerun))
    tasks.append(("cylinder", "single", 0.0, q, q, True, args.dt, args.steps, args.rerun))
    
    # 5. Boomerang: relative yaw angles 0, 45, 90, 180 deg
    for rel_ang in [0, 45, 90, 180]:
        q1 = (1.0, 0.0, 0.0, 0.0)
        q2 = make_quaternion_z(rel_ang)
        for d_r in d_ratios:
            tasks.append(("boomerang", f"rel_{rel_ang}", d_r, q1, q2, False, args.dt, args.steps, args.rerun))
    tasks.append(("boomerang", "single", 0.0, q1, q1, True, args.dt, args.steps, args.rerun))
    
    print(f"\nLaunching {len(tasks)} simulation tasks across {args.workers} workers...")
    t0 = time.time()
    
    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_single_simulation, t): t for t in tasks}
        for f in as_completed(futures):
            res = f.result()
            g = res["geom"]
            tag = res["tag"]
            if g not in all_results:
                all_results[g] = {}
            if tag not in all_results[g]:
                all_results[g][tag] = {}
            all_results[g][tag][str(res["d_r"])] = res
            status_str = f"single" if res["is_single"] else f"d/R={int(res['d_r'])}"
            print(f"  -> Done: {g:10s} [{tag:10s}] {status_str:8s}")
            
    total_time = time.time() - t0
    print(f"\nAll {len(tasks)} simulations finished in {total_time:.2f} seconds!")
    
    # Save combined results summary JSON
    summary_file = os.path.join(data_dir, "trajectories_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary trajectory data saved to: {summary_file}")

if __name__ == "__main__":
    main()
