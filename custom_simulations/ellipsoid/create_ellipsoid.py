import os
import sys

def create_ellipsoid(input_vertex_file, output_vertex_file, a_factor, b_factor, c_factor):
    """
    Reads a sphere .vertex file and stretches it into an ellipsoid.
    a_factor: stretch factor along X axis
    b_factor: stretch factor along Y axis
    c_factor: stretch factor along Z axis
    """
    with open(input_vertex_file, 'r') as f:
        lines = f.readlines()
        
    header = lines[0]
    
    with open(output_vertex_file, 'w') as f:
        f.write(header)
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) == 3:
                x = float(parts[0]) * a_factor
                y = float(parts[1]) * b_factor
                z = float(parts[2]) * c_factor
                f.write(f"{x:.15e}\t{y:.15e}\t{z:.15e}\n")

if __name__ == "__main__":
    base_dir = r"d:\sedimentation_dynamics"
    structures_dir = os.path.join(base_dir, "RigidMultiblobsWall", "multi_bodies", "Structures")
    out_dir = os.path.join(base_dir, "custom_simulations", "ellipsoid")
    
    # We will use the N=162 sphere as our base blueprint
    base_sphere = os.path.join(structures_dir, "shell_N_162_Rg_1_Rh_1_0530.vertex")
    
    # Create a Prolate Ellipsoid (Cigar shape: stretched on Z axis)
    prolate_out = os.path.join(out_dir, "ellipsoid_prolate_N_162.vertex")
    create_ellipsoid(base_sphere, prolate_out, a_factor=1.0, b_factor=1.0, c_factor=2.0)
    print(f"Created {prolate_out} (Prolate: 1x1x2)")
    
    # Create an Oblate Ellipsoid (Pancake shape: squashed on Z axis)
    oblate_out = os.path.join(out_dir, "ellipsoid_oblate_N_162.vertex")
    create_ellipsoid(base_sphere, oblate_out, a_factor=2.0, b_factor=2.0, c_factor=1.0)
    print(f"Created {oblate_out} (Oblate: 2x2x1)")
    
    # Also create the required clones file
    clones_out = os.path.join(out_dir, "single_body.clones")
    with open(clones_out, "w") as f:
        f.write("1\n0 0 0 1 0 0 0\n")
    print(f"Created {clones_out}")
