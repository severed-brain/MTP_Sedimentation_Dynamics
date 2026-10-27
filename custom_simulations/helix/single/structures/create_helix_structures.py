import os
import numpy as np

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
target_dir = os.path.join(base_dir, "custom_simulations", "helix", "single", "structures")
multi_body_struct_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
os.makedirs(target_dir, exist_ok=True)
os.makedirs(multi_body_struct_dir, exist_ok=True)

# Helical geometry parameters
n_turns = 3.0           # 3 complete helical wavelengths
H = 6.6                 # Axial height along Z
R = 0.35                # Helical radius in XY-plane

resolutions = [19, 38, 75]

print("=" * 70)
print("CREATING MULTI-RESOLUTION HELIX STRUCTURES (.vertex)")
print(f"Parameters: Turns={n_turns}, Height H={H}, Radius R={R}")
print("=" * 70)

for N in resolutions:
    t = np.linspace(0.0, 1.0, N)
    z = (t - 0.5) * H
    phi = 2.0 * np.pi * n_turns * (t - 0.5)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    pts = np.column_stack([x, y, z])
    
    # Strictly zero out center of mass to prevent artificial precession
    com = pts.mean(axis=0)
    pts = pts - com
    
    # Compute inter-blob distance and hydrodynamic blob radius
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    avg_d = float(np.mean(dists))
    blob_radius = avg_d / 2.0
    
    vertex_path = os.path.join(target_dir, f"helix_N_{N}.vertex")
    with open(vertex_path, "w") as f:
        f.write(f"# Helix: Turns={n_turns}, H={H}, R={R}, avg_blob_dist={avg_d:.4f}, blob_radius={blob_radius:.4f}\n")
        f.write(f"{N}\n")
        for p in pts:
            f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
            
    print(f"Created: {vertex_path}")
    print(f"  Blobs: N={N}, Avg Spacing: {avg_d:.4f}, Blob Radius a: {blob_radius:.4f}, CoM: {pts.mean(axis=0)}")
    
    # Also save standard N=38 to multi_body structures for pair simulations
    if N == 38:
        mb_vertex_path = os.path.join(multi_body_struct_dir, f"helix_N_{N}.vertex")
        with open(mb_vertex_path, "w") as f:
            f.write(f"# Helix: Turns={n_turns}, H={H}, R={R}, avg_blob_dist={avg_d:.4f}, blob_radius={blob_radius:.4f}\n")
            f.write(f"{N}\n")
            for p in pts:
                f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
        print(f"  Copied N=38 to: {mb_vertex_path}")

print("=" * 70)
