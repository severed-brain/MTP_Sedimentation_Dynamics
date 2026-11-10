import os
import sys
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def get_base_dir():
    d = os.path.abspath(os.path.dirname(__file__))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "RigidMultiblobsWall")):
            return d
        d = os.path.dirname(d)
    return r"d:\sedimentation_dynamics"

base_dir = get_base_dir()
struct_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
plots_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "plots")
thesis_fig_dir = os.path.join(base_dir, "thesis", "figures")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(thesis_fig_dir, exist_ok=True)

def load_vertex(v_name):
    path = os.path.join(struct_dir, v_name)
    verts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(verts)

def rotate_verts_y(verts, angle_deg):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    R_mat = np.array([
        [cos_a, 0.0, sin_a],
        [0.0,   1.0, 0.0  ],
        [-sin_a,0.0, cos_a]
    ])
    return verts @ R_mat.T

def compute_rpy_flow_field(blobs, blob_radius, forces, X_grid, Z_grid, eta=1.0):
    """
    Compute fluid velocity u(x, y=0, z) on a 2D grid using the regularized RPY tensor.
    Vectorized over blobs and grid points.
    """
    grid_shape = X_grid.shape
    X_flat = X_grid.ravel()
    Z_flat = Z_grid.ravel()
    M_points = len(X_flat)
    
    Ux_flat = np.zeros(M_points)
    Uz_flat = np.zeros(M_points)
    
    a = blob_radius
    inv_8pi_eta = 1.0 / (8.0 * math.pi * eta)
    inv_6pi_eta_a = 1.0 / (6.0 * math.pi * eta * a)
    
    for i in range(len(blobs)):
        bx, by, bz = blobs[i]
        fx, fy, fz = forces[i]
        
        # Displacement vector from blob to grid points (at y=0)
        rx = X_flat - bx
        ry = -by
        rz = Z_flat - bz
        
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
        r = np.maximum(r, 1e-12) # avoid div by zero
        
        # Far-field RPY: r >= 2a
        far_mask = (r >= 2.0 * a)
        near_mask = ~far_mask
        
        # Far field
        if np.any(far_mask):
            rf = r[far_mask]
            rf2 = r2[far_mask]
            inv_rf = 1.0 / rf
            inv_rf2 = 1.0 / rf2
            
            # C1 = (1 + 2 a^2 / (3 r^2)) * inv_8pi_eta / r
            # C2 = (1 - 2 a^2 / r^2) * inv_8pi_eta / r^3
            c1 = inv_8pi_eta * inv_rf * (1.0 + (2.0 * a * a / 3.0) * inv_rf2)
            c2 = inv_8pi_eta * inv_rf * inv_rf2 * (1.0 - 2.0 * a * a * inv_rf2)
            
            # dot_product = r . f = rx*fx + ry*fy + rz*fz
            r_dot_f = rx[far_mask] * fx + ry * fy + rz[far_mask] * fz
            
            Ux_flat[far_mask] += c1 * fx + c2 * rx[far_mask] * r_dot_f
            Uz_flat[far_mask] += c1 * fz + c2 * rz[far_mask] * r_dot_f
            
        # Near field: r < 2a
        if np.any(near_mask):
            rn = r[near_mask]
            c1_near = inv_6pi_eta_a * (1.0 - (9.0 * rn) / (32.0 * a))
            c2_near = inv_6pi_eta_a * (3.0 / (32.0 * a * rn))
            
            r_dot_f = rx[near_mask] * fx + ry * fy + rz[near_mask] * fz
            
            Ux_flat[near_mask] += c1_near * fx + c2_near * rx[near_mask] * r_dot_f
            Uz_flat[near_mask] += c1_near * fz + c2_near * rz[near_mask] * r_dot_f
            
    Ux = Ux_flat.reshape(grid_shape)
    Uz = Uz_flat.reshape(grid_shape)
    return Ux, Uz

def main():
    print("=" * 95)
    print("COMPUTING HYDRODYNAMIC FLOW FIELDS (SINGLE VS PAIRWISE) ACROSS GEOMETRIES")
    print("Evaluating Exact RPY Stokesian Flow Fields and Streamlines around Sedimenting Bodies")
    print("=" * 95)
    
    # Grid definition: x in [-6, 6], z in [-6, 6]
    n_pts = 100
    x_vec = np.linspace(-6.0, 6.0, n_pts)
    z_vec = np.linspace(-6.0, 6.0, n_pts)
    X, Z = np.meshgrid(x_vec, z_vec)
    
    shapes_info = [
        {
            "name": "Sphere",
            "v_name": "sphere_N_162.vertex",
            "brad": 0.1380,
            "N": 162,
            "F": 1000.0,
            "d_sep": 3.0,
            "angle": 0.0
        },
        {
            "name": "Prolate Ellipsoid (θ=45° Drift)",
            "v_name": "ellipsoid_prolate_N_162.vertex",
            "brad": 0.1393,
            "N": 162,
            "F": 1000.0,
            "d_sep": 3.0,
            "angle": 45.0
        },
        {
            "name": "Disc (Broadside)",
            "v_name": "disc_N_91.vertex",
            "brad": 0.1000,
            "N": 91,
            "F": 1000.0,
            "d_sep": 3.0,
            "angle": 0.0
        },
        {
            "name": "Cube",
            "v_name": "cube_N_56.vertex",
            "brad": 0.3333,
            "N": 56,
            "F": 1000.0,
            "d_sep": 3.0,
            "angle": 0.0
        }
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 24))
    
    for row_idx, s in enumerate(shapes_info):
        print(f"\nProcessing Flow Field for {s['name']}...")
        raw_verts = load_vertex(s["v_name"])
        rotated_verts = rotate_verts_y(raw_verts, s["angle"])
        N = len(rotated_verts)
        brad = s["brad"]
        f_blob = np.array([0.0, 0.0, -s["F"] / N])
        
        # 1. Single Body at origin (0, 0, 0)
        single_blobs = rotated_verts.copy()
        single_forces = np.tile(f_blob, (N, 1))
        
        Ux_s, Uz_s = compute_rpy_flow_field(single_blobs, brad, single_forces, X, Z)
        speed_s = np.sqrt(Ux_s**2 + Uz_s**2)
        
        # 2. Pairwise Bodies at x = -d/2 and x = +d/2
        d = s["d_sep"]
        b1_blobs = rotated_verts + np.array([-d/2.0, 0.0, 0.0])
        b2_blobs = rotated_verts + np.array([+d/2.0, 0.0, 0.0])
        pair_blobs = np.vstack([b1_blobs, b2_blobs])
        pair_forces = np.tile(f_blob, (2 * N, 1))
        
        Ux_p, Uz_p = compute_rpy_flow_field(pair_blobs, brad, pair_forces, X, Z)
        speed_p = np.sqrt(Ux_p**2 + Uz_p**2)
        
        # Left column: Single Body Flow Field
        ax_left = axes[row_idx, 0]
        cf1 = ax_left.contourf(X, Z, speed_s, levels=40, cmap="viridis", alpha=0.85)
        ax_left.streamplot(X, Z, Ux_s, Uz_s, color="white", linewidth=0.8, density=1.2, arrowsize=1.0)
        # Draw particle blobs near y=0
        mask_y = np.abs(single_blobs[:, 1]) < 0.3
        ax_left.scatter(single_blobs[mask_y, 0], single_blobs[mask_y, 2], s=25, c="red", edgecolors="black", zorder=5)
        ax_left.set_title(f"{s['name']}: Single Particle Wake & Streamlines", fontsize=11, fontweight="bold")
        ax_left.set_xlabel("Lateral Position $x$", fontsize=10)
        ax_left.set_ylabel("Vertical Position $z$", fontsize=10)
        ax_left.set_xlim(-5.5, 5.5)
        ax_left.set_ylim(-5.5, 5.5)
        plt.colorbar(cf1, ax=ax_left, fraction=0.046, pad=0.04, label="Fluid Speed $|u|$")
        
        # Right column: Pairwise Bodies Flow Field
        ax_right = axes[row_idx, 1]
        cf2 = ax_right.contourf(X, Z, speed_p, levels=40, cmap="viridis", alpha=0.85)
        ax_right.streamplot(X, Z, Ux_p, Uz_p, color="white", linewidth=0.8, density=1.2, arrowsize=1.0)
        mask_y1 = np.abs(b1_blobs[:, 1]) < 0.3
        mask_y2 = np.abs(b2_blobs[:, 1]) < 0.3
        ax_right.scatter(b1_blobs[mask_y1, 0], b1_blobs[mask_y1, 2], s=25, c="red", edgecolors="black", zorder=5, label="Body 1")
        ax_right.scatter(b2_blobs[mask_y2, 0], b2_blobs[mask_y2, 2], s=25, c="blue", edgecolors="black", zorder=5, label="Body 2")
        ax_right.set_title(f"{s['name']}: Pair ($d/R = {d:.1f}$) Hydrodynamic Drafting Channel", fontsize=11, fontweight="bold")
        ax_right.set_xlabel("Lateral Position $x$", fontsize=10)
        ax_right.set_ylabel("Vertical Position $z$", fontsize=10)
        ax_right.set_xlim(-5.5, 5.5)
        ax_right.set_ylim(-5.5, 5.5)
        plt.colorbar(cf2, ax=ax_right, fraction=0.046, pad=0.04, label="Fluid Speed $|u|$")
        
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "flow_fields_single_vs_pair_multishape.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated master flow field visualization: {plot_path}")
    
    shutil.copyfile(plot_path, os.path.join(thesis_fig_dir, "flow_fields_single_vs_pair_multishape.png"))
    print(f"Copied to thesis figures: {os.path.join(thesis_fig_dir, 'flow_fields_single_vs_pair_multishape.png')}")
    print("=" * 95)
    print("FLOW FIELDS COMPUTED AND PLOTTED SUCCESSFULLY!")
    print("=" * 95)

if __name__ == "__main__":
    main()
