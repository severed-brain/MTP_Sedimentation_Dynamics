# Master's Thesis Defense Presentation
## Multi-Resolution Sedimentation Dynamics of Rigid Bodies in Viscous Fluids

---

## Slide 1: Title & Author Details
* **Topic:** Hydrodynamic Sedimentation Dynamics of Rigid Bodies in Viscous Fluids
* **Subtopic:** Multi-Resolution Multiblob Boundary Formulation & Numerical Schemes Comparison
* **Candidate:** Thesis Research Group
* **Degree / Program:** Master's Thesis Project (MTP)
* **Framework:** RigidMultiblobsWall (RPY Stokesian Dynamics)

---

## Slide 2: Research Motivation & Problem Statement
* **Low Reynolds Number Regime ($Re \ll 1$):**
  * Inertia is negligible; viscous forces dominate.
  * Reversible, linear, instantaneous dynamics.
* **Limitations of Classical Stokes Law ($F = 6\pi\eta R V$):**
  * Restricted strictly to single, isolated spheres in unbounded domains.
  * Fails for non-spherical bodies (ellipsoids, boomerangs), multi-body clusters, and boundary-confined suspensions.
* **Objective:**
  * Implement and validate a **Rigid Multiblob Method**.
  * Quantify convergence from coarse ($N=12$) to high-resolution ($N=2562$) shells.
  * Compare temporal algorithms: **Adams-Bashforth**, **Forward Euler**, and **Direct Mobility Solve**.
  * Investigate anisotropic drag and orientation-induced lateral drift on ellipsoids.

---

## Slide 3: Mathematical & Hydrodynamic Formulation
* **Governing Stokes System:**
  $$\nabla p = \eta \nabla^2 \mathbf{u}, \quad \nabla \cdot \mathbf{u} = 0$$
* **Rotne-Prager-Yamakawa (RPY) Mobility Tensor:**
  $$\mathbf{u}_i = \sum_{j} \mathbf{M}_{ij} \mathbf{f}_j$$
  * Regularizes singularity of point forces.
  * Strictly Symmetric Positive Definite (SPD) for all blob configurations.
* **Rigid Body Constrained Saddle-Point System:**
  $$\begin{pmatrix} \mathbf{M} & -\mathbf{K} \\ -\mathbf{K}^T & \mathbf{0} \end{pmatrix} \begin{pmatrix} \boldsymbol{\lambda} \\ \mathbf{U} \end{pmatrix} = \begin{pmatrix} \mathbf{0} \\ -\mathbf{F}_{\text{ext}} \end{pmatrix}$$
  * Solved iteratively via Preconditioned GMRES.

---

## Slide 4: Multi-Resolution Shell Discretization
* Spherical shell geometries of hydrodynamic radius $R_h = 1.0$:
  * **$N = 12$ blobs:** $a = 0.5117$
  * **$N = 42$ blobs:** $a = 0.2735$
  * **$N = 162$ blobs:** $a = 0.1392$
  * **$N = 642$ blobs:** $a = 0.0699$
  * **$N = 2562$ blobs:** $a = 0.0350$
* **Optimal Radius Scaling Formula:**
  $$a \approx \frac{1}{2}\sqrt{\frac{4\pi R_g^2}{N}}$$

---

## Slide 5: Computational Complexity & Scaling
* Profiling wall-clock execution time for 5 integration steps:

![Time vs N Plot](figures/time_vs_N_plot.png)

* **Findings:**
  * $N=12 \to 0.07$ s
  * $N=162 \to 0.52$ s
  * $N=642 \to 3.79$ s
  * $N=2562 \to 63.76$ s
* **Execution Command:**
  ```bash
  python custom_simulations/shell/time_complexity_plotter.py
  ```

---

## Slide 6: Multi-Scheme Analysis (Velocity vs. Force)
* Side-by-side comparison across all three schemes:

![Side-by-Side Schemes](figures/schemes_side_by_side_comparison.png)

* Direct overlay and convergence toward Stokes Law:

![Schemes Comparison Overlay](figures/schemes_comparison_vel_vs_force.png)

* **Key Takeaway:**
  * **Adams-Bashforth, Forward Euler, and Mobility agree to $10^{-14}$ precision** in unbounded Stokes flow because steady sedimentation velocity is constant in time ($d\mathbf{V}/dt = 0$).
  * The **Mobility Scheme** requires only **1 solve**, making it $10\times - 20\times$ faster for finding terminal velocity!
  * **$N=2562$ achieves $98.9\%$ agreement** with continuum Stokes Law ($V = 52.45$ vs $53.05$).
* **Execution Command:**
  ```bash
  python custom_simulations/shell/velocity_force_plotter.py --scheme all
  ```

---

## Slide 7: Non-Spherical Particles: Prolate vs. Oblate Ellipsoids
* Investigating particle shape effects on terminal drag:

| Prolate ($1 \times 1 \times 2$) | Oblate ($1 \times 2 \times 2$) |
| :---: | :---: |
| ![Prolate](figures/ellipsoid_velocity_vs_force.png) | ![Oblate](figures/oblate_velocity_vs_force.png) |

* **Prolate:** Lower cross-section along long axis $\implies$ Reduced drag, faster sedimentation.
* **Oblate:** Broad cross-section $\implies$ Elevated drag, slower sedimentation.
* **Execution Commands:**
  ```bash
  python custom_simulations/ellipsoid/ellipsoid_velocity_force_plotter.py
  python custom_simulations/ellipsoid/oblate_velocity_force_plotter.py
  ```

---

## Slide 8: Orientation Coupling & Lateral Drift
* Sedimentation of inclined ellipsoids ($\theta \in [0^\circ, 90^\circ]$) under vertical gravity:

![Lateral Drift](figures/lateral_drift_trajectories.png)

* **Mechanism:**
  * $\theta = 0^\circ$ and $\theta = 90^\circ$: Pure vertical descent (symmetric, $M_{xz} = 0$).
  * $\theta = 45^\circ$: Maximum off-diagonal mobility coupling $\implies$ Strong horizontal drift $V_x \ne 0$.
* **Execution Command:**
  ```bash
  python custom_simulations/ellipsoid_angles/plot_angle_drift.py
  ```

---

## Slide 9: Literature Benchmark Replication (PRE 2024)
* Replicating Figure 3 from *Phys. Rev. E 109, 065302 (2024)*:

![PRE Replication](figures/fig3_replication.png)

* Matches published broad-side to end-on drag ratios across aspect ratios $e \in [0.3, 1.0]$.
* **Execution Command:**
  ```bash
  python custom_simulations/fig3_replication/plot_fig3.py
  ```

---

## Slide 10: Conclusions & Future Research
1. **Convergence Verified:** Multiblob method matches Stokes Law within $1.1\%$ at $N=2562$.
2. **Algorithm Guidelines:**
   * Steady velocity & mobility: Use **Mobility Solve** ($1\times$ solve).
   * Time-dependent trajectories: Use **Adams-Bashforth** ($O(\Delta t^2)$).
3. **Anisotropy & Drift:** Quantified orientation-dependent lateral drift and replicated published PRE benchmark.
4. **Future Scope:** Planar wall interactions (`domain one_wall`), multi-particle hydrodynamic drafting/tumbling, and Brownian fluctuating hydrodynamics.
