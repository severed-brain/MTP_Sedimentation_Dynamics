import os
import numpy as np
import matplotlib.pyplot as plt

# Prolate parameters: a=2, b=1, c=1 (stretched along Z)
a = 2.0
b = 1.0
epsilon = np.sqrt(1 - (b/a)**2)

# Equation 34 from the PhysRevE paper for C_parallel
term1 = -2 * epsilon
term2 = (1 + epsilon**2) * np.log((1 + epsilon) / (1 - epsilon))
C_parallel = (8/3) * (epsilon**3) / (term1 + term2)

# Theoretical velocity (Equation 33)
# F = 6 * pi * eta * a * U_parallel * C_parallel
# => U_parallel = F / (6 * pi * eta * a * C_parallel)
eta = 1.0
def get_theoretical_U(F):
    return F / (6 * np.pi * eta * a * C_parallel)

F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
Ns = [12, 42, 162, 642, 2562]
base_dir = r"d:\sedimentation_dynamics\custom_simulations\ellipsoid"

plt.figure(figsize=(10, 6))

for N in Ns:
    velocities = []
    for F in F_values:
        cfg_file = os.path.join(base_dir, f"N_{N}", "data", f"prolate_N_{N}_F_{int(F)}.single_body.config")
        with open(cfg_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            
        idx = 0
        frames = []
        while idx < len(lines):
            nb = int(lines[idx].split()[0])
            idx += 1
            bodies = []
            for _ in range(nb):
                bodies.append([float(x) for x in lines[idx].split()])
                idx += 1
            frames.append(bodies)
            
        dt = 0.05
        z0 = frames[0][0][2]
        z_end = frames[-1][0][2]
        num_frames = len(frames) - 1
        vz = abs((z_end - z0) / (num_frames * dt))
        velocities.append(vz)
        
    plt.plot(F_values, velocities, marker='o', label=f'Simulation (N = {N})')

# Plot theoretical line
all_forces = np.linspace(0, max(F_values), 100)
theoretical_v = [get_theoretical_U(f) for f in all_forces]
plt.plot(all_forces, theoretical_v, '--', color='black', label='Theoretical (PhysRevE Eq 33/34)')

plt.title('Prolate Ellipsoid Terminal Velocity vs Force (End-on Orientation)')
plt.xlabel('Total Force (N * g)')
plt.ylabel('Terminal Velocity (|V_z|)')
plt.legend()
plt.grid(True)
save_path = os.path.join(base_dir, "ellipsoid_velocity_vs_force.png")
plt.savefig(save_path)
print(f"Successfully generated new plot with theoretical line at {save_path}")
