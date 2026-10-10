import os
import math
import numpy as np

base_dir = r"d:\sedimentation_dynamics"
disc_dir = os.path.join(base_dir, "custom_simulations", "disc")
struct_dir = os.path.join(disc_dir, "structures")
os.makedirs(struct_dir, exist_ok=True)

radius = 1.0
rings_config = {
    19: 2,  # 1 + 6 + 12 = 19
    37: 3,  # 1 + 6 + 12 + 18 = 37
    61: 4,  # 1 + 6 + 12 + 18 + 24 = 61
    91: 5   # 1 + 6 + 12 + 18 + 24 + 30 = 91
}

for N, m in rings_config.items():
    blobs = [[0.0, 0.0, 0.0]]  # central blob
    dr = radius / m
    for ring_idx in range(1, m + 1):
        r = ring_idx * dr
        n_blobs_ring = 6 * ring_idx
        for j in range(n_blobs_ring):
            theta = j * 2.0 * math.pi / n_blobs_ring
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = 0.0
            blobs.append([x, y, z])
            
    assert len(blobs) == N, f"Expected {N} blobs, got {len(blobs)}"
    
    # Save .vertex file
    out_vertex = os.path.join(struct_dir, f"disc_N_{N}.vertex")
    with open(out_vertex, "w") as f:
        f.write(f"{len(blobs)}\n")
        for b in blobs:
            f.write(f"{b[0]:.15e}\t{b[1]:.15e}\t{b[2]:.15e}\n")
            
    optimal_a = radius / (2.0 * m)
    print(f"Created: {out_vertex} | Rings m={m} | N={N} | Optimal blob_radius a={optimal_a:.4f}")
