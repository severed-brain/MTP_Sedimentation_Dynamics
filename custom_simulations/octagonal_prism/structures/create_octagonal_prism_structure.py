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
struct_dir = os.path.join(base_dir, "custom_simulations", "octagonal_prism", "structures")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")
os.makedirs(struct_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

def generate_octagonal_prism(R=1.0, H=2.0, n_intervals=4):
    """
    Generate a surface multiblob regular octagonal prism centered at (0, 0, 0).
    R: circumradius of the regular octagonal base.
    H: height along Z axis.
    n_intervals: number of discrete vertical layers along H.
    """
    # 8 primary vertices on circle of radius R
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8.0
    V = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    
    # 16 boundary points per horizontal cross-section (8 vertices + 8 edge midpoints)
    bnd = []
    for i in range(8):
        p1 = V[i]
        p2 = V[(i + 1) % 8]
        bnd.append(p1)
        bnd.append(0.5 * (p1 + p2))
    bnd = np.array(bnd)
    
    # Inner ring of 8 points at r = 0.5 * R and central point
    inner_ring = 0.5 * bnd[::2]
    center = np.array([[0.0, 0.0]])
    cap_pts = np.vstack([center, inner_ring, bnd]) # 25 points per cap
    
    # Z levels
    z_levels = np.linspace(-H / 2.0, H / 2.0, n_intervals + 1)
    
    all_blobs = []
    # Bottom cap
    for p in cap_pts:
        all_blobs.append([p[0], p[1], z_levels[0]])
    # Top cap
    for p in cap_pts:
        all_blobs.append([p[0], p[1], z_levels[-1]])
    # Intermediate side walls (3 intermediate levels * 16 bnd = 48 points)
    for z in z_levels[1:-1]:
        for p in bnd:
            all_blobs.append([p[0], p[1], z])
            
    pts = np.array(all_blobs, dtype=np.float64)
    # Ensure exact zero center of mass
    pts -= pts.mean(axis=0)
    
    # Blob radius (touching spheres condition: mean spacing = 0.385 -> a = 0.1925)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(pts))
    np.fill_diagonal(D, np.inf)
    mean_spacing = float(np.mean(np.min(D, axis=1)))
    blob_radius = round(mean_spacing / 2.0, 4)
    
    return pts, blob_radius

def save_vertex_file(pts, blob_radius, filename):
    filepath = os.path.join(struct_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"# Regular Octagonal Prism: N={len(pts)}, blob_radius={blob_radius:.4f}\n")
        f.write(f"{len(pts)}\n")
        for p in pts:
            f.write(f"{p[0]:.15e}\t{p[1]:.15e}\t{p[2]:.15e}\n")
    print(f"Saved: {filepath} (N = {len(pts)}, blob_radius = {blob_radius:.4f})")
    return filepath

def main():
    print("=" * 80)
    print("GENERATING MULTIBLOB REGULAR OCTAGONAL PRISM STRUCTURE (N=98)")
    print("=" * 80)
    
    pts, brad = generate_octagonal_prism(R=1.0, H=2.0, n_intervals=4)
    fname = "octagonal_prism_N_98.vertex"
    save_vertex_file(pts, brad, fname)
    
    # 3D Visualization
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis',
                   s=140, edgecolors='black', linewidth=1.0, alpha=0.9)
    
    ax.set_title(f"3D Regular Octagonal Prism ($N={len(pts)}$, $a={brad:.4f}$, $H=2.0$, $R=1.0$)\n$D_{{8h}}$ Symmetry Group",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X (Width)", fontsize=11, labelpad=8)
    ax.set_ylabel("Y (Depth)", fontsize=11, labelpad=8)
    ax.set_zlabel("Z (Height)", fontsize=11, labelpad=8)
    
    cbar = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Z Coordinate (Height Level)", fontsize=10)
    
    ax.set_box_aspect([1, 1, 2])
    ax.view_init(elev=25, azim=40)
    plt.tight_layout()
    
    plot_path = os.path.join(struct_dir, "octagonal_prism_structure_3d.png")
    thesis_plot_path = os.path.join(thesis_fig_dir, "octagonal_prism_structure_3d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(thesis_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D render: {plot_path}")

if __name__ == "__main__":
    main()
