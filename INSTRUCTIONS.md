# RigidMultiblobsWall — Execution & Continuation Guide

This document summarizes the exact steps, file locations, CLI commands, and physical background needed to run and extend the simulations in this project.

---

## 1. Directory & File Organization

* **Repository Core**: `RigidMultiblobsWall/`
* **Solver Scripts**: `RigidMultiblobsWall/multi_bodies/`
  * `multi_bodies.py`: Dynamic time integrator engine.
  * `multi_bodies_utilities.py`: Static mobility / resistance solver.
  * `Structures/`: Geometry definitions (`.vertex`) and initial positions (`.clones`).
  * `data/`: Simulation outputs (`.config`, `.bodies_info`, `.time`, etc.).
* **Driver Script**: `simulate_shell_sedimentation.py` (Workspace root).

---

## 2. Available Shell Discretizations (`Structures/`)

| Discretization | Vertex File | Clones File | Number of Blobs $N$ |
| :---: | :--- | :--- | :---: |
| $N = 12$ | `Structures/shell_N_12_Rg_1_Rh_1_2625.vertex` | `Structures/shell_N_12_Rg_1.clones` | 12 |
| $N = 42$ | `Structures/shell_N_42_Rg_1_Rh_1_1220.vertex` | `Structures/shell_N_12_Rg_1.clones` | 42 |
| $N = 162$ | `Structures/shell_N_162_Rg_1_Rh_1_0530.vertex` | `Structures/shell_N_12_Rg_1.clones` | 162 |
| $N = 642$ | `Structures/shell_N_642_Rg_1_Rh_1_0239.vertex` | `Structures/shell_N_12_Rg_1.clones` | 642 |
| $N = 2562$ | `Structures/shell_N_2562_Rg_1_Rh_1_0113.vertex` | `Structures/shell_N_12_Rg_1.clones` | 2562 |

---

## 3. How to Run Simulations

### Option A: Using the Repository's Native Command-Line Interface
Navigate to `multi_bodies` and run with the corresponding input file:

```bash
cd c:\MTP-oct\RigidMultiblobsWall\multi_bodies

# N = 12
python multi_bodies.py --input-file inputfile_shell_N_12_free_fall.dat

# N = 42
python multi_bodies.py --input-file inputfile_shell_N_42_free_fall.dat

# N = 162
python multi_bodies.py --input-file inputfile_shell_N_162_free_fall.dat

# N = 642
python multi_bodies.py --input-file inputfile_shell_N_642_free_fall.dat

# N = 2562
python multi_bodies.py --input-file inputfile_shell_N_2562_free_fall.dat
```

### Option B: Using the Python Driver Script
From `c:\MTP-oct`:

```bash
# Run all resolutions (12, 42, 162, 642) and print convergence table:
python simulate_shell_sedimentation.py

# Run for a specific N only:
python simulate_shell_sedimentation.py --N 42
```

---

## 4. How the Physics is Configured

In the input files (`inputfile_shell_N_*_free_fall.dat`), free fall (unbounded fluid, no wall) is set via:

```text
scheme                                   deterministic_adams_bashforth
domain                                   no_wall
mobility_blobs_implementation            python_no_wall
mobility_vector_prod_implementation      python_no_wall
blob_blob_force_implementation           None
dt                                       0.05
n_steps                                  20
n_save                                   1
eta                                      1.0
g                                        1.0
blob_radius                              0.25
repulsion_strength                       0.0
debye_length                             1.0
repulsion_strength_wall                  0.0
debye_length_wall                        1.0
save_clones                              one_file
output_name                              data/shell_N_<N>_free_fall
structure Structures/shell_N_<N>_...vertex Structures/shell_N_12_Rg_1.clones
```

* **`domain no_wall`**: Instructs the solver to use the Rotne-Prager-Yamakawa (RPY) mobility tensor instead of Blake wall corrections, and deactivates wall-boundary checks ($z > 0$).
* **`repulsion_strength_wall 0.0`**: Sets wall potential to zero.
* **`blob_blob_force_implementation None`**: Simulates pure Stokes flow without artificial inter-blob springs.

---

## 5. Convergence Results

| $N$ | Time Steps | Terminal Velocity $V_z$ (units/s) | $\frac{V_z}{N \cdot g}$ (Normalized Mobility) |
| :---: | :---: | :---: | :---: |
| **12** | 20 | $-0.70662$ | $-0.058885$ |
| **42** | 20 | $-2.25267$ | $-0.053635$ |
| **162** | 20 | $-8.60135$ | $-0.053095$ |
| **642** | 20 | $-34.03432$ | $-0.053013$ |
| **2562** | 5 | $-135.79034$ | $-0.053002$ |

---

## 6. Custom Simulations & Best Practices

To maintain integrity, **never modify the `RigidMultiblobsWall/` core repository.** Treat it as a read-only source of truth. All custom tests and new shape experiments should be isolated in a separate `custom_simulations/` directory.

### Directory Structure for Custom Tests
Organize your custom tests by shape and blob resolution ($N$):
```text
custom_simulations/
└── <shape_name>/ (e.g., shell, cylinder, boomerang)
    └── N_<size>/ (e.g., N_162)
        ├── data/                         # Simulation output goes here
        ├── input.dat                     # The master configuration script
        ├── single_body.clones            # Map of positions (1 body)
        └── my_custom_shape.vertex        # Geometry blueprint
```

### The Three Essential Files
Every rigid body simulation requires exactly three files:
1. **`.vertex` (The Blueprint):** Defines the shape of a single object by listing the $X, Y, Z$ coordinates of every blob that makes it up. The first line specifies the number of blobs $N$.
2. **`.clones` (The Map):** Defines **how many** bodies are in the fluid and **where** they start (Center of Mass $X, Y, Z$ and quaternions). It is completely independent of $N$. E.g., a file starting with `1` places exactly 1 body.
3. **`.dat` (The Master Script):** The configuration file executed by the physics engine. It sets all physical parameters (viscosity, time steps, blob radius) and uses the `structure` line at the bottom to glue the `.vertex` and `.clones` files together. Always copy parameters exactly from the source-of-truth repo without assuming new physics values.

---

## 7. Next Steps for Continuing the Project
1. **Near-Wall vs. Free-Fall Comparison**: Run the same series with `domain single_wall` to show wall drag slowing down sedimentation.
2. **Brownian Dynamics**: Set `kT > 0` and scheme to `stochastic_Slip_Trapz` to observe diffusion alongside sedimentation.
3. **Articulated Bodies**: Reference `multi_bodies/examples/bacteria` to link bodies with passive/active joint constraints (`.const`).
4. **Active Particles**: Reference `multi_bodies/examples/squirmer` to add active slip surface boundary conditions (`.slip`).
