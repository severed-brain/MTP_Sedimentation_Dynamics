import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"c:\MTP-oct"

base_dir = get_base_dir()
struct_dir = os.path.join(base_dir, "custom_simulations", "hexagonal_prism", "structures")
os.makedirs(struct_dir, exist_ok=True)

def generate_hexagonal_prism(k_s=2, R=1.0, H=2.0):
    """
    Generate a surface multiblob hexagonal prism centered at (0, 0, 0).
    k_s: discretization order of the regular hexagonal base (number of segments per edge).
    R: circumradius of the regular hexagonal cross section (edge length s = R).
    H: height along Z axis.
    """
    delta = R / k_s
    
    # 1. Generate 2D regular hexagonal grid for end-caps
    cap_pts = []
    boundary_indices = []
    idx = 0
    for q in range(-k_s, k_s + 1):
        r1 = max(-k_s, -q - k_s)
        r2 = min(k_s, -q + k_s)
        for r in range(r1, r2 + 1):
            x = (q + r * 0.5) * delta
            y = (r * np.sqrt(3) / 2.0) * delta
            cap_pts.append([x, y])
            if max(abs(q), abs(r), abs(q + r)) == k_s:
                boundary_indices.append(idx)
            idx += 1
            
    cap_pts = np.array(cap_pts)
    bnd_pts_2d = cap_pts[boundary_indices]
    
    # Sort boundary points in circular order
    angles = np.arctan2(bnd_pts_2d[:, 1], bnd_pts_2d[:, 0])
    bnd_pts_2d = bnd_pts_2d[np.argsort(angles)]
    
    # 2. Discrete Z levels along height
    n_intervals = int(round(H / delta))
    z_levels = np.linspace(-H / 2.0, H / 2.0, n_intervals + 1)
    
    all_blobs = []
    # Bottom cap at z_levels[0]
    for p in cap_pts:
        all_blobs.append([p[0], p[1], z_levels[0]])
    # Top cap at z_levels[-1]
    for p in cap_pts:
        all_blobs.append([p[0], p[1], z_levels[-1]])
    # 6 side rectangular faces (intermediate z levels)
    for z in z_levels[1:-1]:
        for p in bnd_pts_2d:
            all_blobs.append([p[0], p[1], z])
            
    pts = np.array(all_blobs, dtype=np.float64)
    # Ensure exact zero center of mass
    pts -= pts.mean(axis=0)
    
    # Hydrodynamic blob radius (touching spheres condition: 2a = delta)
    blob_radius = delta / 2.0
    return pts, blob_radius

def save_vertex_file(pts, blob_radius, filename):
    filepath = os.path.join(struct_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"# Hexagonal Prism: N={len(pts)}, blob_radius={blob_radius:.4f}\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
    print(f"Saved: {filepath} (N = {len(pts)}, blob_radius = {blob_radius:.4f})")
    return filepath

def main():
    print("=" * 80)
    print("GENERATING MULTIBLOB HEXAGONAL PRISM STRUCTURE")
    print("=" * 80)
    
    # Standard high-quality single-resolution discretization: k_s=2, R=1.0, H=2.0 -> N=74
    pts, brad = generate_hexagonal_prism(k_s=2, R=1.0, H=2.0)
    fname = "hexagonal_prism_N_74.vertex"
    save_vertex_file(pts, brad, fname)
    
    # 3D Visualization
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='coolwarm',
                   s=180, edgecolors='black', linewidth=1.2, alpha=0.9)
    
    ax.set_title(f"Regular Hexagonal Prism ($N={len(pts)}$, $a={brad:.3f}$, $H=2.0$, $R=1.0$)",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X (Width)", fontsize=11, labelpad=10)
    ax.set_ylabel("Y (Width)", fontsize=11, labelpad=10)
    ax.set_zlabel("Z (Height)", fontsize=11, labelpad=10)
    
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Z Coordinate", fontsize=10)
    
    plt.tight_layout()
    viz_path = os.path.join(struct_dir, "hexagonal_prism_structure_3d.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated 3D visualization: {viz_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

