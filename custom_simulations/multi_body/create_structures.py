import os
import math
import shutil

base_dir = r"d:\sedimentation_dynamics"
mb_structures = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies", "Structures")
target_dir = os.path.join(base_dir, "custom_simulations", "multi_body", "structures")
os.makedirs(target_dir, exist_ok=True)

# 1. Sphere (N=42)
src_sphere = os.path.join(mb_structures, "shell_N_42_Rg_1_Rh_1_1220.vertex")
dst_sphere = os.path.join(target_dir, "sphere_N_42.vertex")
shutil.copy(src_sphere, dst_sphere)
print(f"Created: {dst_sphere}")

# 2. Ellipsoid (Prolate N=42, 1x1x2)
dst_ellipsoid = os.path.join(target_dir, "ellipsoid_prolate_N_42.vertex")
with open(src_sphere, "r") as f:
    lines = f.readlines()
with open(dst_ellipsoid, "w") as f:
    f.write(lines[0])
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            x = float(parts[0]) * 1.0
            y = float(parts[1]) * 1.0
            z = float(parts[2]) * 2.0
            f.write(f"{x:.15e}\t{y:.15e}\t{z:.15e}\n")
print(f"Created: {dst_ellipsoid}")

# 3. Disc (Planar circle, N=37, radius R=1.0)
dst_disc = os.path.join(target_dir, "disc_N_37.vertex")
blobs = [[0.0, 0.0, 0.0]]
radius = 1.0
# Ring 1
for i in range(6):
    th = i * 2 * math.pi / 6
    blobs.append([0.33 * radius * math.cos(th), 0.33 * radius * math.sin(th), 0.0])
# Ring 2
for i in range(12):
    th = i * 2 * math.pi / 12
    blobs.append([0.67 * radius * math.cos(th), 0.67 * radius * math.sin(th), 0.0])
# Ring 3
for i in range(18):
    th = i * 2 * math.pi / 18
    blobs.append([1.0 * radius * math.cos(th), 1.0 * radius * math.sin(th), 0.0])

with open(dst_disc, "w") as f:
    f.write(f"{len(blobs)}\n")
    for b in blobs:
        f.write(f"{b[0]:.15e}\t{b[1]:.15e}\t{b[2]:.15e}\n")
print(f"Created: {dst_disc} (N={len(blobs)})")

# 4. Cylinder (N=86)
src_cyl = os.path.join(mb_structures, "Cylinder_N_86_Lg_1_9384_Rg_0_1484.vertex")
dst_cyl = os.path.join(target_dir, "cylinder_N_86.vertex")
shutil.copy(src_cyl, dst_cyl)
print(f"Created: {dst_cyl}")

# 5. Boomerang (N=15)
src_boom = os.path.join(mb_structures, "boomerang_N_15.vertex")
dst_boom = os.path.join(target_dir, "boomerang_N_15.vertex")
shutil.copy(src_boom, dst_boom)
print(f"Created: {dst_boom}")
