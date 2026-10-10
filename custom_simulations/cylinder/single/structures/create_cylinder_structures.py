import os
import shutil
import math
import numpy as np

base_dir = r"d:\sedimentation_dynamics"
mb_structures = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies", "Structures")
target_dir = os.path.join(base_dir, "custom_simulations", "cylinder", "single", "structures")
os.makedirs(target_dir, exist_ok=True)

# 1. N = 14 (centerline rod)
src_14 = os.path.join(mb_structures, "Cylinder_N_14_Lg_1_9295_Rg_0_18323.vertex")
dst_14 = os.path.join(target_dir, "cylinder_N_14.vertex")
shutil.copyfile(src_14, dst_14)
print(f"Created: {dst_14} (N=14)")

# 2. N = 44 (7 rings of 6 blobs + 2 endcaps)
# Generate from N=86 geometry: length L=1.9384, Rg=0.1484, 7 rings along x
dst_44 = os.path.join(target_dir, "cylinder_N_44.vertex")
blobs_44 = []
x_vals_7 = np.linspace(-0.9692, 0.9692, 7)
rg = 0.1484
for x in x_vals_7:
    for k in range(6):
        th = k * 2.0 * math.pi / 6.0
        blobs_44.append([x, rg * math.sin(th), rg * math.cos(th)])
# 2 endcaps
blobs_44.append([-0.9692, 0.0, 0.0])
blobs_44.append([0.9692, 0.0, 0.0])
with open(dst_44, "w") as f:
    f.write(f"{len(blobs_44)}\n")
    for b in blobs_44:
        f.write(f"{b[0]:.15e}\t{b[1]:.15e}\t{b[2]:.15e}\n")
print(f"Created: {dst_44} (N={len(blobs_44)})")

# 3. N = 86 (14 rings of 6 blobs + 2 endcaps)
src_86 = os.path.join(mb_structures, "Cylinder_N_86_Lg_1_9384_Rg_0_1484.vertex")
dst_86 = os.path.join(target_dir, "cylinder_N_86.vertex")
shutil.copyfile(src_86, dst_86)
print(f"Created: {dst_86} (N=86)")

# 4. N = 324 (26 rings of 12 blobs + 12 endcaps)
src_324 = os.path.join(mb_structures, "Cylinder_N_324_Lg_2_0299_Rg_0_1554.vertex")
dst_324 = os.path.join(target_dir, "cylinder_N_324.vertex")
shutil.copyfile(src_324, dst_324)
print(f"Created: {dst_324} (N=324)")
