# Thesis & Presentation Repository

This directory contains the formal academic thesis documentation, presentation slide deck, and generated figures for the Master's Thesis Project (MTP) on **Hydrodynamic Sedimentation Dynamics of Rigid Multiblobs**.

---

## Directory Contents

```
thesis/
├── THESIS.md          # Complete, publication-ready academic thesis document (Chapters 1–9)
├── PRESENTATION.md    # Defense presentation deck with embedded slides and speaker notes
├── README.md          # Overview and execution guide for the thesis folder
└── figures/           # High-resolution publication figures & plots
    ├── schemes_side_by_side_comparison.png   # Multi-scheme side-by-side comparison
    ├── schemes_comparison_vel_vs_force.png   # Overlay and resolution convergence
    ├── time_vs_N_plot.png                    # Wall-clock scaling vs blob count N
    ├── adams_bashforth_vel_vs_force.png      # Adams-Bashforth individual plot
    ├── forward_euler_vel_vs_force.png        # Forward Euler individual plot
    ├── mobility_vel_vs_force.png             # Mobility direct solve individual plot
    ├── ellipsoid_velocity_vs_force.png       # Prolate ellipsoid velocity vs force
    ├── oblate_velocity_vs_force.png          # Oblate ellipsoid velocity vs force
    ├── lateral_drift_trajectories.png        # Orientation coupling and cross-stream drift
    └── fig3_replication.png                  # Literature benchmark replication (PRE 2024)
```

---

## Quick Execution Guide

All figures and numerical data can be generated and reproduced using the following commands:

```bash
# 1. Multi-Scheme Comparison (Adams-Bashforth, Forward Euler, Mobility):
python custom_simulations/shell/velocity_force_plotter.py --scheme all

# 2. Computational Time Complexity vs Resolution N:
python custom_simulations/shell/time_complexity_plotter.py

# 3. Anisotropic Ellipsoids (Prolate & Oblate):
python custom_simulations/ellipsoid/ellipsoid_velocity_force_plotter.py
python custom_simulations/ellipsoid/oblate_velocity_force_plotter.py

# 4. Inclination Angle & Lateral Drift:
python custom_simulations/ellipsoid_angles/plot_angle_drift.py

# 5. Published Literature Replication (Phys. Rev. E 109, 065302):
python custom_simulations/fig3_replication/plot_fig3.py
```
