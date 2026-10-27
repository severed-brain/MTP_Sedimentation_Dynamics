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
struct_dir = os.path.join(base_dir, "custom_simulations", "trefoil_knot", "structures")
os.makedirs(struct_dir, exist_ok=True)

def generate_trefoil_knot(N=64, scale_xy=0.5, scale_z=0.8):
    """
    Generate a 3D chiral trefoil knot T(2,3) multiblob structure centered at (0, 0, 0).
    N: number of constituent blobs along the knotted closed filament loop.
    scale_xy, scale_z: spatial scaling factors to ensure non-overlapping strand separation.
    """
    t = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    
    # Parametric equations for a right-handed trefoil knot
    x = scale_xy * (np.sin(t) + 2.0 * np.sin(2.0 * t))
    y = scale_xy * (np.cos(t) - 2.0 * np.cos(2.0 * t))
    z = scale_z * (-np.sin(3.0 * t))
    
    pts = np.column_stack([x, y, z])
    # Strictly zero out center of mass
    pts -= pts.mean(axis=0)
    
    # Compute inter-blob distance along the loop
    diffs = np.roll(pts, -1, axis=0) - pts
    dists = np.linalg.norm(diffs, axis=1)
    avg_d = float(np.mean(dists))
    blob_radius = avg_d / 2.0
    
    return pts, blob_radius, avg_d

def save_vertex_file(pts, blob_radius, filename):
    filepath = os.path.join(struct_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"# Chiral Trefoil Knot T(2,3): N={len(pts)}, blob_radius={blob_radius:.4f}\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
    print(f"Saved: {filepath} (N = {len(pts)}, blob_radius = {blob_radius:.4f})")
    return filepath

def main():
    print("=" * 80)
    print("GENERATING COMPLEX MULTIBLOB CHIRAL TREFOIL KNOT STRUCTURE")
    print("=" * 80)
    
    N = 64
    pts, brad, avg_d = generate_trefoil_knot(N=N, scale_xy=0.5, scale_z=0.8)
    fname = f"trefoil_knot_N_{N}.vertex"
    save_vertex_file(pts, brad, fname)
    
    # 3D Visualization
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Draw tube backbone line
    loop_pts = np.vstack([pts, pts[0]])
    ax.plot(loop_pts[:, 0], loop_pts[:, 1], loop_pts[:, 2], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Scatter blobs
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='plasma',
                   s=190, edgecolors='black', linewidth=1.2, alpha=0.92)
    
    ax.set_title(f"3D Chiral Trefoil Knot $T(2,3)$ ($N={N}$, $a={brad:.4f}$)\nTopological Rigid Particle in Stokes Flow",
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("X (Span)", fontsize=11, labelpad=10)
    ax.set_ylabel("Y (Span)", fontsize=11, labelpad=10)
    ax.set_zlabel("Z (Height)", fontsize=11, labelpad=10)
    
    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-1.6, 1.6])
    ax.set_zlim([-1.2, 1.2])
    
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Z Coordinate (Elevation)", fontsize=10)
    
    plt.tight_layout()
    viz_path = os.path.join(struct_dir, "trefoil_knot_structure_3d.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated 3D visualization: {viz_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
