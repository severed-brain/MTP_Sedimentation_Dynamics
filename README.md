# Sedimentation Dynamics & Hydrodynamic Coupling (RigidMultiblobsWall)

This repository contains the simulation framework, driver automation scripts, configuration files, and theoretical documentation for the study of **Sedimentation Dynamics of Porous & Rigid Shells in Stokes Flow** using the **Rigid Multiblobs Wall (RMBW)** method (Usabiaga et al.).

---

## 🔬 Project Overview

In low-Reynolds-number hydrodynamics (Stokes flow), multiblob methods discretize rigid particle surfaces into constrained spheres ("blobs") governed by hydrodynamic mobility tensors. This repository analyzes:
1. **Free-Fall Sedimentation & Convergence**: Validating the hydrodynamic drafting and shielding of discretized spherical shells across resolutions ($N \in \{12, 42, 162, 642, 2562\}$).
2. **Mobility Scaling**: Computing normalized terminal velocity $\frac{V_z}{N \cdot g}$ and observing monotonic convergence toward the theoretical continuum limit ($\approx -0.05300\,\text{units/s}$).
3. **Hydrodynamic Coupling near Boundaries**: Preparing and analyzing near-wall hydrodynamic resistance corrections via Blake tensor formulations.

---

## 📁 Repository Structure

```text
MTP_Sedimentation_Dynamics/
├── RigidMultiblobsWall/           # RMBW simulation package (Usabiaga et al.)
│   ├── multi_bodies/              # Core time-integrator and mobility solvers
│   │   ├── multi_bodies.py        # Main solver executable
│   │   ├── Structures/            # Discretized geometry (.vertex) & clones (.clones)
│   │   ├── inputfile_shell_*.dat  # Shell configuration scripts
│   │   └── data/                  # Benchmark logs and trajectory outputs
│   ├── articulated/               # Joint & linkage solvers
│   ├── mobility/                  # RPY and wall mobility implementations
│   └── visit/                     # Visualization export utilities
├── simulations/                   # Simulation directories and output benchmarks
├── simulate_shell_sedimentation.py# Master driver script for automated runs
├── mtp.md                         # In-depth theoretical monograph & logbook
├── INSTRUCTIONS.md                # Step-by-step CLI execution manual
├── .gitignore                     # Git ignore rules
└── README.md                      # Project overview & documentation
```

---

## ⚡ Quickstart

### Prerequisites
* Python 3.8+
* `numpy`
* `scipy`

Install dependencies:
```bash
pip install numpy scipy
```

### Running Simulations

#### Option 1: Automated Multi-Resolution Driver
Execute the automated batch runner from the workspace root:

```bash
# Run all discretizations (N = 12, 42, 162, 642) and print convergence table
python simulate_shell_sedimentation.py

# Run for a specific resolution (e.g. N = 42)
python simulate_shell_sedimentation.py --N 42
```

#### Option 2: Native RMBW Solver CLI
Run directly using `multi_bodies.py` inside `RigidMultiblobsWall/multi_bodies/`:

```bash
cd RigidMultiblobsWall/multi_bodies

# Example: Shell N = 12
python multi_bodies.py --input-file inputfile_shell_N_12_free_fall.dat

# Example: Shell N = 42
python multi_bodies.py --input-file inputfile_shell_N_42_free_fall.dat
```

---

## 📊 Benchmark Convergence Results

Sedimentation of spherical shells under gravity ($g = 1.0$) in unbounded Stokes flow ($\eta = 1.0$):

| Shell Discretization ($N$) | Number of Blobs | Integration Steps | Terminal Velocity $V_z$ (units/s) | Normalized Mobility $\frac{V_z}{N \cdot g}$ |
| :---: | :---: | :---: | :---: | :---: |
| **$N = 12$** | 12 | 20 | $-0.70662$ | $-0.058885$ |
| **$N = 42$** | 42 | 20 | $-2.25267$ | $-0.053635$ |
| **$N = 162$** | 162 | 20 | $-8.60135$ | $-0.053095$ |
| **$N = 642$** | 642 | 20 | $-34.03432$ | $-0.053013$ |
| **$N = 2562$** | 2562 | 5 | $-135.79034$ | $-0.053002$ |

> **Convergence Note**: As blob density increases, hydrodynamic shielding becomes continuous, asymptotically approaching the normalized mobility $-0.05300\,\text{units/s}$.

---

## 📖 Documentation & References

* [INSTRUCTIONS.md](INSTRUCTIONS.md) — Step-by-step CLI and workflow documentation.
* [mtp.md](mtp.md) — Comprehensive research monograph, mathematical formulation, and experiment logbook.
* **RigidMultiblobsWall Reference**: F. Balboa Usabiaga, B. Delmotte, and D. Donev, *Hydrodynamics of suspensions of passive and active rigid bodies*, J. Chem. Phys. 146, 134104 (2017).
