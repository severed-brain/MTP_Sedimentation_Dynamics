import os
import shutil
import numpy as np

base_dir = r"d:\sedimentation_dynamics"
mb_structures = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies", "Structures")
target_dir = os.path.join(base_dir, "custom_simulations", "boomerang", "single", "structures")
os.makedirs(target_dir, exist_ok=True)

# 1. Boomerang N = 7 (apex + 3 blobs along each arm)
dst_7 = os.path.join(target_dir, "boomerang_N_7.vertex")
blobs_7 = [[0.0, 0.0, 0.0]] # apex
s_7 = 2.1 / 3.0 # 0.70
for i in range(1, 4):
    blobs_7.append([i * s_7, 0.0, 0.0]) # arm 1 along +x
for i in range(1, 4):
    blobs_7.append([0.0, i * s_7, 0.0]) # arm 2 along +y

with open(dst_7, "w") as f:
    f.write(f"{len(blobs_7)}\n")
    for b in blobs_7:
        f.write(f"{b[0]:.15e}\t{b[1]:.15e}\t{b[2]:.15e}\n")
print(f"Created: {dst_7} (N={len(blobs_7)})")

# 2. Boomerang N = 11 (apex + 5 blobs along each arm)
dst_11 = os.path.join(target_dir, "boomerang_N_11.vertex")
blobs_11 = [[0.0, 0.0, 0.0]] # apex
s_11 = 2.1 / 5.0 # 0.42
for i in range(1, 6):
    blobs_11.append([i * s_11, 0.0, 0.0])
for i in range(1, 6):
    blobs_11.append([0.0, i * s_11, 0.0])

with open(dst_11, "w") as f:
    f.write(f"{len(blobs_11)}\n")
    for b in blobs_11:
        f.write(f"{b[0]:.15e}\t{b[1]:.15e}\t{b[2]:.15e}\n")
print(f"Created: {dst_11} (N={len(blobs_11)})")

# 3. Boomerang N = 15 (apex + 7 blobs along each arm)
src_15 = os.path.join(mb_structures, "boomerang_N_15.vertex")
dst_15 = os.path.join(target_dir, "boomerang_N_15.vertex")
shutil.copyfile(src_15, dst_15)
print(f"Created: {dst_15} (N=15)")
