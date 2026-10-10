import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"d:\sedimentation_dynamics"
shell_dir = os.path.join(base_dir, "custom_simulations", "shell")

Ns = [12, 42, 162, 642, 2562]
F_values = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
schemes = ["adams_bashforth", "forward_euler", "mobility"]
scheme_titles = {
    "adams_bashforth": "Adams-Bashforth (2nd Order)",
    "forward_euler": "Forward Euler (1st Order)",
    "mobility": "Mobility (Quasi-Steady Direct)"
}

def parse_velocity_adams_bashforth(n_dir, N, F):
    cfg_file = os.path.join(n_dir, "data", f"shell_N_{N}_F_{int(F)}.single_body.config")
    if not os.path.exists(cfg_file):
        return None
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
    return abs((z_end - z0) / (num_frames * dt))

def parse_velocity_forward_euler(n_dir, N, F):
    return parse_velocity_adams_bashforth(n_dir, N, F)

def parse_velocity_mobility(n_dir, N, F):
    vel_file = os.path.join(n_dir, "data", f"shell_N_{N}_F_{int(F)}.velocity.dat")
    if not os.path.exists(vel_file):
        return None
    vel_data = np.loadtxt(vel_file)
    if vel_data.ndim == 1:
        return abs(vel_data[2])
    return abs(vel_data[0, 2])

parsers = {
    "adams_bashforth": parse_velocity_adams_bashforth,
    "forward_euler": parse_velocity_forward_euler,
    "mobility": parse_velocity_mobility
}

# Collect all data
results = {s: {N: {} for N in Ns} for s in schemes}

for scheme in schemes:
    scheme_dir = os.path.join(shell_dir, scheme)
    parser = parsers[scheme]
    for N in Ns:
        n_dir = os.path.join(scheme_dir, f"N_{N}")
        for F in F_values:
            v = parser(n_dir, N, F)
            if v is not None:
                results[scheme][N][F] = v

# Plotting: Figure with 3 subplots side-by-side for each scheme + Figure with convergence overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

all_forces = np.linspace(0, max(F_values), 100)
theoretical_v = all_forces / (6 * np.pi * 1.0 * 1.0)
colors = {12: "#1f77b4", 42: "#ff7f0e", 162: "#2ca02c", 642: "#d62728", 2562: "#9467bd"}

for ax_idx, scheme in enumerate(schemes):
    ax = axes[ax_idx]
    for N in Ns:
        forces = []
        vels = []
        for F in F_values:
            if F in results[scheme][N]:
                forces.append(F)
                vels.append(results[scheme][N][F])
        if forces:
            ax.plot(forces, vels, marker='o', label=f'N = {N}', color=colors[N], linewidth=1.8, markersize=5)
            
    ax.plot(all_forces, theoretical_v, '--', color='black', label='Stokes Law (Theoretical)', linewidth=1.5)
    ax.set_title(scheme_titles[scheme], fontsize=13, fontweight='bold')
    ax.set_xlabel('Total Force (N · g)', fontsize=11)
    if ax_idx == 0:
        ax.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
fig_path = os.path.join(shell_dir, "schemes_side_by_side_comparison.png")
plt.savefig(fig_path, dpi=300)
print(f"Side-by-side comparison saved to: {fig_path}")

# Figure 2: Direct Overlay & Convergence comparison
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Overlay of all schemes
markers = {"adams_bashforth": "o", "forward_euler": "s", "mobility": "^"}
linestyles = {"adams_bashforth": "-", "forward_euler": "--", "mobility": ":"}

for N in Ns:
    for scheme in schemes:
        forces = [F for F in F_values if F in results[scheme][N]]
        vels = [results[scheme][N][F] for F in forces]
        if forces:
            label = f'N={N} ({scheme[:3].upper()})' if N in [12, 2562] else None
            ax1.plot(forces, vels, marker=markers[scheme], linestyle=linestyles[scheme],
                     color=colors[N], label=label, alpha=0.85, markersize=5)

ax1.plot(all_forces, theoretical_v, '--', color='black', linewidth=2.0, label='Stokes Law')
ax1.set_title('Direct Overlay: Adams-Bashforth vs Euler vs Mobility', fontsize=12, fontweight='bold')
ax1.set_xlabel('Total Force (N · g)', fontsize=11)
ax1.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(fontsize=9)

# Panel B: Convergence vs N at F=1000
ref_F = 1000.0
stokes_ref = ref_F / (6 * np.pi)

for scheme in schemes:
    ns_available = [N for N in Ns if ref_F in results[scheme][N]]
    vels_at_ref = [results[scheme][N][ref_F] for N in ns_available]
    if ns_available:
        ax2.plot(ns_available, vels_at_ref, marker=markers[scheme], linestyle='-',
                 label=scheme_titles[scheme], linewidth=1.8, markersize=7)

ax2.axhline(stokes_ref, color='black', linestyle='--', linewidth=1.5, label=f'Stokes Law ({stokes_ref:.2f})')
ax2.set_xscale('log')
ax2.set_title(f'Velocity Convergence vs Resolution N (Force = {ref_F:.0f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Discretization Resolution (N blobs)', fontsize=11)
ax2.set_ylabel('Terminal Velocity (|V_z|)', fontsize=11)
ax2.set_xticks(Ns)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
ax2.legend(fontsize=9)

plt.tight_layout()
fig2_path = os.path.join(shell_dir, "schemes_comparison_vel_vs_force.png")
plt.savefig(fig2_path, dpi=300)
print(f"Overlay & convergence comparison saved to: {fig2_path}")

# Print summary table
print("\n" + "="*80)
print(f"{'N':>6} | {'Force':>7} | {'Adams-Bashforth':>16} | {'Forward Euler':>14} | {'Mobility':>10} | {'Stokes Law':>10}")
print("="*80)
for N in Ns:
    for F in [1000.0, 3000.0, 5000.0]:
        v_ab = results['adams_bashforth'][N].get(F, float('nan'))
        v_fe = results['forward_euler'][N].get(F, float('nan'))
        v_mob = results['mobility'][N].get(F, float('nan'))
        v_th = F / (6 * np.pi)
        print(f"{N:6d} | {F:7.0f} | {v_ab:16.4f} | {v_fe:14.4f} | {v_mob:10.4f} | {v_th:10.4f}")
print("="*80)
