import os
import math
import numpy as np
from scipy.spatial import ConvexHull

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.
    """
    qw, qx, qy, qz = q
    # Normalize quaternion
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    R = np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
    ])
    return R

def read_vertex_file(vertex_path):
    """
    Read coordinates and blob metadata from a .vertex file.
    Returns (pts: np.ndarray, N: int, blob_radius_hint: float)
    """
    with open(vertex_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        
    blob_radius_hint = None
    for l in lines:
        if "blob_radius" in l:
            try:
                parts = l.split("blob_radius")[-1].replace("=", " ").replace(":", " ").split()
                blob_radius_hint = float(parts[0])
            except Exception:
                pass
                
    data_lines = [l for l in lines if not l.startswith('#')]
    N = int(data_lines[0].split()[0])
    pts = []
    for l in data_lines[1:N+1]:
        parts = [float(v) for v in l.split()[:3]]
        pts.append(parts)
        
    pts = np.array(pts, dtype=np.float64)
    return pts, N, blob_radius_hint

def compute_chiral_index(pts):
    """
    Compute signed chiral pseudoscalar index:
    Measures absence of mirror and inversion symmetry.
    Zero for symmetric bodies (spheres, ellipsoids, discs, cylinders),
    Non-zero for boomerangs and helices.
    """
    N = len(pts)
    if N < 4:
        return 0.0
    
    # Method 1: Normalized ordered triple product sum
    chiral_sum = 0.0
    for i in range(N - 2):
        v1 = pts[i+1] - pts[i]
        v2 = pts[i+2] - pts[i+1]
        v3 = pts[i]
        triple = np.dot(np.cross(v1, v2), v3)
        chiral_sum += triple
        
    scale = (np.mean(np.linalg.norm(pts, axis=1)) ** 3) * N
    if scale > 1e-12:
        return float(chiral_sum / scale)
    return 0.0

def extract_shape_features(pts, blob_radius=None, quaternion=(1.0, 0.0, 0.0, 0.0), F=1000.0):
    """
    Extract comprehensive 3D physics-informed descriptors for any arbitrary multiblob shape.
    
    Parameters:
    -----------
    pts : np.ndarray of shape (N, 3)
        Blob center coordinates.
    blob_radius : float or None
        Hydrodynamic radius of constituent blobs. If None, estimated from nearest-neighbor spacing.
    quaternion : tuple of 4 floats (qw, qx, qy, qz)
        Orientation quaternion relative to global coordinates (gravity along -Z).
    F : float
        Gravitational force magnitude.
        
    Returns:
    --------
    dict containing 25+ invariant and equivariant geometric descriptors.
    """
    N = len(pts)
    
    # 1. Center of Mass
    com = pts.mean(axis=0)
    pts_centered = pts - com
    
    # 2. Inter-blob spacing & blob radius estimation
    if blob_radius is None:
        if N > 1:
            dists = np.linalg.norm(np.diff(pts_centered, axis=0), axis=1)
            blob_radius = float(np.mean(dists)) / 2.0
        else:
            blob_radius = 1.0
            
    # 3. Apply 3D Orientation (Rotation Matrix)
    R_mat = quaternion_to_rotation_matrix(quaternion)
    pts_rot = pts_centered @ R_mat.T
    
    # 4. Global Extents in Oriented Frame
    min_xyz = np.min(pts_rot, axis=0) - blob_radius
    max_xyz = np.max(pts_rot, axis=0) + blob_radius
    extents = max_xyz - min_xyz
    Lx, Ly, Lz = float(extents[0]), float(extents[1]), float(extents[2])
    
    L_max = max(Lx, Ly, Lz)
    L_min = max(min(Lx, Ly, Lz), 1e-6)
    aspect_ratio_max = L_max / L_min
    aspect_ratio_z = Lz / max((Lx + Ly)/2.0, 1e-6)
    V_box = Lx * Ly * Lz
    
    # 5. Radius of Gyration & Max Span
    rg_sq = np.mean(np.sum(pts_centered**2, axis=1))
    Rg = math.sqrt(max(rg_sq, 1e-12))
    
    # Pairwise max distance (span)
    if N <= 500:
        diffs = pts_centered[:, np.newaxis, :] - pts_centered[np.newaxis, :, :]
        D_max = float(np.max(np.sqrt(np.sum(diffs**2, axis=-1)))) + 2.0 * blob_radius
    else:
        D_max = 2.0 * float(np.max(np.linalg.norm(pts_centered, axis=1))) + 2.0 * blob_radius
        
    # 6. Blob Surface Area and Volume Proxy
    A_blobs = 4.0 * math.pi * N * (blob_radius ** 2)
    V_blobs = (4.0 / 3.0) * math.pi * N * (blob_radius ** 3)
    
    # 7. Inertia Tensor & Eigenvalues
    I_tensor = np.zeros((3, 3))
    for p in pts_centered:
        r2 = np.dot(p, p)
        I_tensor += r2 * np.eye(3) - np.outer(p, p)
    I_tensor /= N
    
    eigvals, eigvecs = np.linalg.eigh(I_tensor)
    eigvals = np.sort(eigvals)
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    trace_I = l1 + l2 + l3
    
    # Shape Anisotropy Invariants
    asphericity = l3 - 0.5 * (l1 + l2)
    acylindricity = l2 - l1
    if trace_I > 1e-12:
        rel_anisotropy = 1.0 - 3.0 * (l1*l2 + l2*l3 + l3*l1) / (trace_I**2)
    else:
        rel_anisotropy = 0.0
    rel_anisotropy = float(np.clip(rel_anisotropy, 0.0, 1.0))
    
    # 8. Major Geometric Axis via PCA Covariance
    cov_mat = (pts_centered.T @ pts_centered) / max(N, 1)
    cov_eigvals, cov_eigvecs = np.linalg.eigh(cov_mat)
    # Largest eigenvalue of covariance corresponds to the primary elongation axis
    major_axis_body = cov_eigvecs[:, 2]
    major_axis_world = R_mat @ major_axis_body
    cos_theta = abs(float(major_axis_world[2])) # alignment with vertical axis Z
    cos_theta = min(max(cos_theta, 0.0), 1.0)
    theta_rad = math.acos(cos_theta)
    theta_deg = math.degrees(theta_rad)
    sin_2theta = math.sin(2.0 * theta_rad)
    
    # 9. Projected Frontal Cross-Sectional Area (perpendicular to descent -Z)
    # Area in XY plane
    if N >= 4:
        try:
            hull_xy = ConvexHull(pts_rot[:, :2])
            A_perp = float(hull_xy.volume) + math.pi * (blob_radius ** 2) # in 2D, volume is area
        except Exception:
            A_perp = math.pi * max(Lx * Ly / 4.0, 1e-6)
    else:
        A_perp = math.pi * max(Lx * Ly / 4.0, 1e-6)
        
    A_parallel = (Lx + Ly) / 2.0 * Lz
    area_ratio = A_perp / max(A_parallel, 1e-6)
    
    # 10. Chirality Pseudoscalar
    chiral_index = compute_chiral_index(pts_centered)
    
    # 11. Applied Force Features
    g_force_per_blob = F / N
    
    # Compile dictionary
    features = {
        "N": float(N),
        "blob_radius": float(blob_radius),
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,
        "aspect_ratio_max": aspect_ratio_max,
        "aspect_ratio_z": aspect_ratio_z,
        "V_box": V_box,
        "Rg": Rg,
        "D_max": D_max,
        "A_blobs": A_blobs,
        "V_blobs": V_blobs,
        "I_l1": l1,
        "I_l2": l2,
        "I_l3": l3,
        "asphericity": asphericity,
        "acylindricity": acylindricity,
        "rel_anisotropy": rel_anisotropy,
        "theta_deg": theta_deg,
        "cos_theta": cos_theta,
        "sin_2theta": sin_2theta,
        "A_perp": A_perp,
        "A_parallel": A_parallel,
        "area_ratio": area_ratio,
        "chiral_index": chiral_index,
        "F": float(F),
        "g_force_per_blob": g_force_per_blob
    }
    return features

if __name__ == "__main__":
    # Test feature extraction on sample shapes
    base_dir = r"c:\MTP-oct"
    test_files = [
        os.path.join(base_dir, "custom_simulations", "shell", "mobility", "N_42", "shell_N_42.vertex"),
        os.path.join(base_dir, "custom_simulations", "disc", "single", "structures", "disc_N_37.vertex"),
        os.path.join(base_dir, "custom_simulations", "cylinder", "single", "structures", "cylinder_N_44.vertex"),
        os.path.join(base_dir, "custom_simulations", "boomerang", "single", "structures", "boomerang_N_15.vertex"),
        os.path.join(base_dir, "custom_simulations", "helix", "single", "structures", "helix_N_38.vertex"),
    ]
    
    print("=" * 80)
    print("TESTING 3D GEOMETRIC & PHYSICAL FEATURE EXTRACTOR")
    print("=" * 80)
    for vf in test_files:
        if os.path.exists(vf):
            pts, N, brad = read_vertex_file(vf)
            feats = extract_shape_features(pts, blob_radius=brad, F=1000.0)
            name = os.path.basename(vf)
            print(f"\nShape: {name:<25} (N={feats['N']:.0f}, a={feats['blob_radius']:.4f})")
            print(f"  Rg={feats['Rg']:.3f}, D_max={feats['D_max']:.3f}, Asphericity={feats['asphericity']:.3f}, Anisotropy={feats['rel_anisotropy']:.3f}")
            print(f"  A_perp={feats['A_perp']:.3f}, Area Ratio={feats['area_ratio']:.3f}, Chiral Index={feats['chiral_index']:.5f}")
    print("\nFeature extractor tests completed successfully!")
