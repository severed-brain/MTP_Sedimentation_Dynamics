import os
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
struct_dir = os.path.join(base_dir, "custom_simulations", "cube", "structures")
os.makedirs(struct_dir, exist_ok=True)

def generate_surface_cube(k_points_per_edge=5, half_side=1.0):
    """
    Generate a surface multiblob cube centered at (0, 0, 0).
    k_points_per_edge: number of blobs along each edge of the cube.
    half_side: half-length of cube side (side length L = 2 * half_side).
    """
    coords_1d = np.linspace(-half_side, half_side, k_points_per_edge)
    blobs = []
    eps = 1e-6
    
    for x in coords_1d:
        for y in coords_1d:
            for z in coords_1d:
                # Keep points on the 6 outer boundary faces
                if (abs(abs(x) - half_side) < eps or 
                    abs(abs(y) - half_side) < eps or 
                    abs(abs(z) - half_side) < eps):
                    blobs.append([x, y, z])
                    
    pts = np.array(blobs, dtype=np.float64)
    # Ensure exact zero center of mass
    pts -= pts.mean(axis=0)
    
    # Calculate spacing and optimal hydrodynamic radius
    step = 2.0 * half_side / (k_points_per_edge - 1)
    blob_radius = step / 2.0
    
    return pts, blob_radius

def save_vertex_file(pts, blob_radius, filename):
    filepath = os.path.join(struct_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
    print(f"Saved: {filepath} (N = {len(pts)}, blob_radius = {blob_radius:.4f})")
    return filepath

def main():
    print("=" * 80)
    print("GENERATING MULTIBLOB CUBE STRUCTURES")
    print("=" * 80)
    
    resolutions = [
        (4, "cube_N_56.vertex"),
        (5, "cube_N_98.vertex"),
        (6, "cube_N_152.vertex")
    ]
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (k, fname) in enumerate(resolutions):
        pts, brad = generate_surface_cube(k_points_per_edge=k, half_side=1.0)
        save_vertex_file(pts, brad, fname)
        
        # Plot 3D visualization
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        # Scatter points
        p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', 
                       s=150, edgecolors='black', alpha=0.85)
        ax.set_title(f"Cube $N={len(pts)}$ ($k={k}$, $a={brad:.3f}$)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        
    plt.tight_layout()
    viz_path = os.path.join(struct_dir, "cube_structures_3d.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated 3D visualization: {viz_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
