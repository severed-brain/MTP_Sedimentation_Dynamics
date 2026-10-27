import os
import sys
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
mb_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies")
cube_dir = os.path.join(base_dir, "custom_simulations", "cube")
struct_dir = os.path.join(cube_dir, "structures")
ang_dir = os.path.join(cube_dir, "angles")
data_dir = os.path.join(ang_dir, "data")
plots_dir = os.path.join(ang_dir, "plots")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = f"{mb_dir};{os.path.join(base_dir, 'RigidMultiblobsWall')}"

def run_angle_solve(vertex_file, blob_radius, angle_deg, F_total=2000.0):
    with open(vertex_file, "r") as f:
        N = int(f.readline().strip())
    g_val = F_total / N
    
    ang_rad = math.radians(angle_deg)
    qw = math.cos(ang_rad / 2.0)
    qy = math.sin(ang_rad / 2.0)
    
    clones_name = f"cube_ang_{int(angle_deg)}.clones"
    clones_path = os.path.join(ang_dir, clones_name)
    with open(clones_path, "w") as f:
        f.write(f"1\n0.0 0.0 0.0 {qw:.8f} 0.0 {qy:.8f} 0.0\n")
        
    out_prefix = f"data/cube_N_{N}_ang_{int(angle_deg)}"
    inp_content = f"""scheme mobility
mobility_blobs_implementation python_no_wall
mobility_vector_prod_implementation python_no_wall
eta 1.0
g {g_val:.8f}
blob_radius {blob_radius:.6f}
repulsion_strength 0.0
debye_length 1.0
repulsion_strength_wall 0.0
debye_length_wall 1.0
output_name {out_prefix}
structure {os.path.basename(vertex_file)} {clones_name}
"""
    inp_file = os.path.join(ang_dir, f"input_ang_{int(angle_deg)}.dat")
    with open(inp_file, "w") as f:
        f.write(inp_content)
        
    import shutil
    v_target = os.path.join(ang_dir, os.path.basename(vertex_file))
    if not os.path.exists(v_target):
        shutil.copyfile(vertex_file, v_target)
        
    cmd = [sys.executable, os.path.join(mb_dir, "multi_bodies_utilities.py"), "--input-file", os.path.basename(inp_file)]
    subprocess.run(cmd, cwd=ang_dir, env=env, check=True, capture_output=True)
    
    vel_file = os.path.join(ang_dir, f"{out_prefix}.velocity.dat")
    vel = np.loadtxt(vel_file)
    return {
        "angle": angle_deg,
        "Vx": abs(float(vel[0])),
        "Vy": abs(float(vel[1])),
        "Vz": abs(float(vel[2])),
        "Wx": abs(float(vel[3])),
        "Wy": abs(float(vel[4])),
        "Wz": abs(float(vel[5])),
        "glide_ratio": abs(float(vel[0]) / float(vel[2])) if abs(float(vel[2])) > 1e-6 else 0.0
    }

def main():
    print("=" * 80)
    print("STOKESIAN SIMULATION: CUBE ORIENTATION ANGLE SWEEP (N=98, F=2000)")
    print("=" * 80)
    
    v_file = os.path.join(struct_dir, "cube_N_98.vertex")
    blob_radius = 0.25
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    records = []
    
    print(f"{'Angle (deg)':<12} {'Vz (units/s)':<16} {'Vx (units/s)':<16} {'Glide |Vx/Vz|':<16}")
    print("-" * 60)
    
    for ang in angles:
        res = run_angle_solve(v_file, blob_radius, ang, F_total=2000.0)
        records.append(res)
        print(f"{ang:<12.1f} {res['Vz']:<16.4f} {res['Vx']:<16.4e} {res['glide_ratio']:<16.4e}")
        
    # Save CSV
    out_csv = os.path.join(data_dir, "cube_angle_sweep.csv")
    with open(out_csv, "w") as f:
        f.write("angle_deg,Vz,Vx,Vy,Wz,glide_ratio\n")
        for r in records:
            f.write(f"{r['angle']},{r['Vz']},{r['Vx']},{r['Vy']},{r['Wz']},{r['glide_ratio']}\n")
    print(f"\nSaved angle sweep data to: {out_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
