# Master's Thesis Defense Presentation
## Multi-Resolution Sedimentation Dynamics of Rigid Bodies in Viscous Fluids

---

## Slide 1: Title & Author Details
* **Topic:** Hydrodynamic Sedimentation Dynamics of Rigid Bodies in Viscous Fluids
* **Subtopic:** Multi-Resolution Multiblob Boundary Formulation & Numerical Schemes Comparison
* **Candidate:** Thesis Research Group (Department of Mechanical Engineering)
* **Degree / Program:** Master's Thesis Project (MTP)
* **Framework:** RigidMultiblobsWall (RPY Stokesian Dynamics, Blake Tensor, Saddle-Point GMRES)

---

## Slide 2: Motivation & Real-World Sedimentation Applications
* **Why Study Idealized Stokes Sedimentation?**
  * Provides the fundamental, exact hydrodynamic baseline for multi-body drafting and wake interactions before adding multiphysics complexities.
* **Industrial Application: Inkjet Printing (Continuous & Drop-on-Demand):**
  * Suspended pigment colloids sediment and draft, causing aggregation and nozzle clogging ($10 - 30\,\mu\text{m}$).
  * High-frequency tandem droplet ejection: viscous wake reduces drag on trailing droplets $\implies$ trailing acceleration, satellite droplet merging, and impact splattering.
* **Geophysical Transport: Marine Snow & Oceanic Carbon Pump:**
  * Organic aggregates sediment to the ocean floor, sequestering gigatons of carbon.
  * Recent breakthrough by Rahul Chajwa & Manu Prakash (Science 2024): microscopic flow fields and "comet tails" alter sinking speed and global carbon models.
* **Biomedical & Microfluidic Diagnostics:**
  * Erythrocyte Sedimentation Rate (ESR) clinical inflammation assays; microfluidic passive cell sorting (CTCs, leukocytes).

---

## Slide 3: Hydrodynamic Derivation: Non-Dimensionalization ($Re \to 0$)
* **Full Incompressible Navier-Stokes Equations:**
  $$\rho \left( \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} \right) = -\nabla p + \eta \nabla^2 \mathbf{u} + \mathbf{f}_{\text{ext}}, \quad \nabla \cdot \mathbf{u} = 0$$
* **Normalization via Characteristic Scales:**
  * Length: $L_c = R_h$, Velocity: $U_c = V_0$, Time: $t_c = L_c/U_c$, Viscous Pressure: $p_c = \eta U_c / L_c$.
  * Dimensionless variables: $\mathbf{x}^* = \mathbf{x}/L_c$, $\mathbf{u}^* = \mathbf{u}/U_c$, $p^* = (p - p_\infty)/p_c$, $\nabla^* = L_c \nabla$.
* **Dimensionless Momentum Equation:**
  $$\boxed{ Re \left( \frac{\partial \mathbf{u}^*}{\partial t^*} + (\mathbf{u}^* \cdot \nabla^*) \mathbf{u}^* \right) = -\nabla^* p^* + \nabla^{*2} \mathbf{u}^* + \mathbf{f}^*_{\text{ext}} }, \quad Re = \frac{\rho U_c L_c}{\eta}$$
* **Vanishing of Inertia as $Re \to 0$ ($Re \ll 1$):**
  * Left-hand side vanishes completely $\implies$ **Linear, Instantaneous Stokes Flow**:
    $$\eta \nabla^2 \mathbf{u} - \nabla p + \mathbf{f}_{\text{ext}} = \mathbf{0}, \quad \nabla \cdot \mathbf{u} = 0$$
  * **Key Properties:**
    1. *Strict Linearity:* Exact linear superposition of velocity wakes.
    2. *Instantaneity:* Zero fluid memory; flow adapts instantaneously to boundaries.
    3. *Kinematic Reversibility:* Purcell's scallop theorem; reversible streamlines.

---

## Slide 4: Green's Function Formalism: Why and How?
* **What is a Green's Function in Stokes Flow?**
  * The fundamental velocity response at point $\mathbf{x}$ to a concentrated point force $\mathbf{F} \delta(\mathbf{x} - \mathbf{x}_0)$:
    $$\eta \nabla^2 \mathbf{u} - \nabla p = -\mathbf{F} \delta(\mathbf{x} - \mathbf{x}_0), \quad \nabla \cdot \mathbf{u} = 0$$
* **Why Use Green's Functions?**
  * Bypasses full 3D volumetric mesh generation (FEM/FVM).
  * Reduces 3D fluid volume PDEs to 2D boundary surface integrals (or 0D discrete point systems):
    $$\mathbf{u}(\mathbf{x}) = \int_{\partial \Omega} \mathbf{G}(\mathbf{x}, \mathbf{y}) \cdot \mathbf{f}(\mathbf{y}) \, dS_y$$
* **Free-Space vs. Wall-Bounded Green's Functions:**
  * **Free-Space Oseen Tensor (Stokeslet):**
    $$\mathbf{G}(\mathbf{r}) = \frac{1}{8\pi\eta r} \left( \mathbf{I} + \frac{\mathbf{r}\mathbf{r}^T}{r^2} \right) \sim \mathcal{O}\left(\frac{1}{r}\right)$$
  * **Blake Tensor (Planar Wall at $z = 0$):** Image system (Stokeslet + inverted image Stokeslet + Stokes-dipole + potential doublet) strictly enforcing $\mathbf{u}(x, y, 0) = \mathbf{0}$.
* **Far-Field vs. Near-Field & Multiblob Discretization:**
  * *Far Field ($r \gg R$):* Any body (sphere, disc, cylinder) collapses to a point Stokeslet.
  * *Near Field ($r \sim R$):* Point-force singularity regularized via Rotne-Prager-Yamakawa (RPY) tensor; rigid kinematics enforced via saddle-point GMRES.

---

## Slide 5: Computational Architecture & RigidMultiblobsWall Framework
* **Software Heritage & Authoritative Source:**
  * Framework: `RigidMultiblobsWall` (Balboa Usabiaga et al., Courant Institute / Flatiron Institute / CAMCoS 2016, JCP 2017).
  * High-performance Stokesian dynamics framework for passive, active, and multi-body suspensions.
* **Core Module Breakdown:**
  * `mobility/mobility.py`: Free-space RPY and wall-bounded Blake/RPB tensor evaluations; block-diagonal preconditioned GMRES saddle-point solver.
  * `multi_bodies/multi_bodies.py`: Dynamic time integrator (Forward Euler, Adams-Bashforth, stochastic RFD).
  * `multi_bodies/multi_bodies_utilities.py`: Direct quasi-steady mobility solver (`scheme mobility`, `scheme body_mobility`).
  * `quaternion_integrator/`: Norm-preserving orientation tracking on $S^3$ unit hypersphere: $\frac{d\mathbf{q}}{dt} = \frac{1}{2}\mathbf{\Omega} \odot \mathbf{q}$.
* **Standardized File Formats:**
  * `*.vertex`: Relative blob coordinates $(x_i, y_i, z_i)$ and hydrodynamic radius $a_i$ in body frame.
  * `*.clones`: Global center-of-mass coordinates $(x, y, z)$ and orientation unit quaternions $(q_0, q_1, q_2, q_3)$.
  * `input.dat`: Problem scheme, domain (`free_space` vs `one_wall`), viscosity $\eta$, timestep $dt$, and solver tolerance ($10^{-8}$).
* **Telemetry & Automation Layer:**
  * Outputs: `*.velocities.dat` ($6$-DOF velocities), `*.body_mobility.dat` ($6\times 6$ tensor), `*.clones` (trajectory snapshots), `*.time` (wallclock runtime).
  * Automation: Custom Python wrappers orchestrate sweeps ($d/R \in [2.0, 12.0]$), format whitespace-delimited scientific `.dat` tables, and generate 4-panel trajectory plots.

---

## Slide 6: Multi-Resolution Shell Discretization & Convergence
* Discretizing spherical shells ($R_h = 1.0$) into $N$ regularized blobs:
  * **$N = 12$ blobs:** $a = 0.5117$
  * **$N = 42$ blobs:** $a = 0.2735$
  * **$N = 162$ blobs:** $a = 0.1392$
  * **$N = 642$ blobs:** $a = 0.0699$
  * **$N = 2562$ blobs:** $a = 0.0350$
* **Optimal Radius Scaling Formula:**
  $$a \approx \frac{1}{2}\sqrt{\frac{4\pi R_g^2}{N}}$$
* **Continuum Stokes Agreement:** $N = 2562$ achieves $98.9\%$ agreement with Stokes Law ($V = 52.45$ vs $53.05$).

---

## Slide 7: Computational Complexity & Scheme Evaluation
* Execution wall-clock time for 5 integration steps:
  * $N=12 \to 0.07$ s
  * $N=162 \to 0.52$ s
  * $N=642 \to 3.79$ s
  * $N=2562 \to 63.76$ s
* **Scheme Comparison (Adams-Bashforth vs. Forward Euler vs. Mobility):**
  * Identical terminal velocity to $10^{-14}$ precision because acceleration is zero ($d\mathbf{V}/dt = 0$).
  * **Direct Mobility Scheme** requires only **1 solve**, making it $10\times - 20\times$ faster for finding terminal velocity and mobility matrices!

---

## Slide 8: Non-Spherical Particles: Prolate vs. Oblate Ellipsoids
* Shape anisotropy modulates terminal drag and settling speed:
  * **Prolate ($1 \times 1 \times 2$):** Slender profile along settling axis $\implies$ Reduced drag, faster sedimentation ($V_0 = 40.45$).
  * **Oblate ($1 \times 2 \times 2$):** Broad frontal area $\implies$ Elevated drag, slower sedimentation ($V_0 = 33.10$).
* **Orientation Coupling & Lateral Drift:**
  * Inclined ellipsoids ($\theta \in [0^\circ, 90^\circ]$) exhibit strong off-diagonal mobility coupling ($M_{xz} \ne 0$), reaching peak lateral drift at $\theta = 45^\circ$.
  * Validated against benchmark literature (*Phys. Rev. E 109, 065302, 2024*).

---

## Slide 9: Multi-Body Pair Hydrodynamic Drafting & Orientation Coupling
* **Master Normalized Scaling Across 9 Geometries ($V_z / V_0$ vs. $d/R$):**
  * Normalizing by solitary terminal baseline $V_0$ collapses all shapes onto continuum Stokeslet behavior.
  * **Spherical Shells ($N = 162$):** $V_z / V_0 = 1.4236$ at $d/R = 2.0$, decaying monotonically to $1.0796$ at $d/R = 10.0$ ($0.43\%$ Stokeslet deviation).
  * **Circular Discs ($N = 91$, Broadside):** $V_z / V_0 = 1.7053$ ($+70.5\%$ speedup), closely tracking broadside Stokeslet $1 + \frac{2}{\pi}\frac{R}{d}$.
  * **Cylinders ($N = 86$, Broadside):** $V_z / V_0 = 1.6014$ ($+60.1\%$ speedup).
  * **Regular Cubes ($N = 56$):** $V_z / V_0 = 1.5449$ ($+54.5\%$ speedup at $d/R = 2.0$).
  * **Chiral Helices ($N = 38$):** $V_z / V_0 = 1.0693$ at $d/R = 4.0$.
  * **Prolate Ellipsoids ($N = 162$):** Orientation-dependent drafting:
    * $\theta = 0^\circ$ (Vertical/Slender, $V_0 = 54.33$ u/s): $V_z / V_0 = 1.3608$ at $d/R = 2.0$.
    * $\theta = 45^\circ$ (Oblique Glider, $V_0 = 47.81$ u/s): Strong lateral drift ($V_x = 8.52$ u/s, drift angle $\phi \approx 10.1^\circ$).
    * $\theta = 90^\circ$ (Broadside, $V_0 = 42.14$ u/s): Maximal frontal area, $V_z / V_0 = 1.4120$.
* **High-Resolution Asymmetric Angle Pairs ($\theta_1 \ne \theta_2$):**
  * In mismatched pairs ($(0^\circ, 45^\circ), (45^\circ, 90^\circ), (0^\circ, 90^\circ)$), differential drag creates persistent relative velocities $\Delta V_z = |V_z^{(1)} - V_z^{(2)}| \approx 3.3 - 12.3\,\text{u/s}$, driving overtaking dynamics.
* **Key Figures:** `master_normalized_pair_scaling.png`, `highres_ellipsoid_angle_pairs.png`, `ellipsoid_angle_trajectories_0_45_90.png`.

---

## Slide 10: Collective Sedimentation & The Crowley Instability
* **5-Body Linear Horizontal Array ($M = 5$):**
  * Symmetrical flanks focus downward momentum at the array center.
  * **Center-to-Edge Velocity Gradients ($V_{\text{center}} / V_{\text{edge}}$):**
    * Spheres ($N=42$): $+27.5\%$ peak at $d/R = 3.0$
    * Discs ($N=37$): $+36.2\%$ peak at $d/R = 3.0$
    * Cylinders ($N=86$): $+37.2\%$ peak at $d/R = 3.0$
    * Ellipsoids ($N=42$): $+24.8\%$ peak at $d/R = 4.0$
    * Regular Cubes ($N=56$): $+21.8\%$ peak at $d/R = 4.0$ ($V_z / V_0 = 2.5341$ at $d/R=2$)
    * Chiral Helices ($N=38$): $+21.6\%$ peak at $d/R = 3.0$ ($V_z / V_0 = 1.5380$ at $d/R=2$)
  * **Crowley Instability Mechanism (Crowley 1971):**
    * Additive wake acceleration pulls center bodies down $\implies$ lateral fluid convergence draws neighbors inward $\implies$ runaway sagging into inverted-U / chevron profiles, triggering droplet pinch-off.
* **5-Body Pentagonal Ring Cluster ($M = 5$):**
  * Rotational symmetry prevents sagging; accelerates entire suspension collectively:
    * Regular Cubes: $V_z / V_0 = 2.8239$ ($+182.4\%$ speedup at $d/R = 2.0$)
    * Circular Discs: $V_z / V_0 = 2.4968$ ($+149.7\%$ speedup)
    * Spherical Shells: $V_z / V_0 = 2.4403$ ($+144.0\%$ speedup)
    * Prolate Ellipsoids: $V_z / V_0 = 2.3479$ ($+134.8\%$ speedup)
    * Chiral Helices: $V_z / V_0 = 1.7411$ ($+74.1\%$ speedup)
* **Key Figures:** `sphere_5body_dynamics.png`, `cube_5body_dynamics.png`, `helix_5body_dynamics.png`.

---

## Slide 11: Multi-Body Rotational Dynamics & Hydrodynamic Rolling
* **Physical Mechanism of Induced Rolling:**
  * Velocity shear across the downward wake ($\partial u_z / \partial x$) exerts hydrodynamic torque on neighboring bodies.
  * Body 1 ($x < 0$) rolls clockwise ($\Omega_y > 0$); Body 2 ($x > 0$) rolls counter-clockwise ($\Omega_y < 0$).
  * **Strict Gear-Mesh Symmetry:** $\Omega_y^{(1)}(d/R) \equiv -\Omega_y^{(2)}(d/R)$ verified across all geometries.
* **Fundamental Power-Law Scaling Proof:**
  * Translational drafting decays as **monopole**: $V_z/V_0 - 1 \propto (R/d)^1$.
  * Induced angular velocity decays as **dipole**: $\Omega_y \propto \frac{V_0}{R}(R/d)^2$.
  * Log-log regression confirms exact **$-2.0$ slope** across Spheres, Discs, Cubes, and Cylinders!
* **Chiral Auto-Rotation & Coupled Spin:**
  * Solitary chiral helices execute intrinsic auto-rotation $\Omega_0 = -11.91\,\text{rad/s}$ due to translation-rotation coupling ($B \ne 0$).
  * In pair interactions, hydrodynamic screening slightly retards spin ($\Omega_z = -11.71\,\text{rad/s}$ at $d/R = 2.0$) while superimposing mutual rolling $\Omega_y = +1.92\,\text{rad/s}$.
* **Key Figure:** `multi_body_rotational_dynamics.png`.

---

## Slide 12: Exact Hydrodynamic Flow Fields & Drafting Corridors
* **Direct RPY Green's Function Streamline Evaluation:**
  * Computed exact velocity vector field $\mathbf{u}(x, y=0, z)$ and speed heatmap $|u|$ across Spheres, Tilted Ellipsoids ($\theta=45^\circ$), Discs, and Cubes.
* **Solitary Particle Wake:**
  * Downward momentum jet decaying as $1/r$.
  * For tilted ellipsoid ($\theta=45^\circ$), wake is deflected laterally, providing the reaction force for oblique gliding.
* **Pairwise Drafting Corridor ($d = 3.0 R$):**
  * Wakes superimpose constructively in the central gap between bodies.
  * Entrained fluid moves downward at high speed, creating a low-resistance corridor that reduces viscous form drag and accelerates settling by $+30\%$ to $+54\%$.
* **Key Figure:** `flow_fields_single_vs_pair_multishape.png`.

---

## Slide 13: Non-Spherical Interactions & Keplerian Orbits (Rahul Chajwa's Work)
* **Spheres vs. Anisotropic Discs:**
  * Spheres have reflection symmetry $\implies$ fall along straight vertical paths at constant distance ($d = \text{const}$).
  * Anisotropic bodies (discs, oblate spheroids) couple **orientation to translation** ($V_{\text{drift}} \propto \sin 2\theta$).
* **Rahul Chajwa et al. (PRL 2019): "Kepler Orbits in Pairs of Disks":**
  * Wake of Disc 1 exerts torque on Disc 2 $\implies$ tilts Disc 2 $\implies$ induces lateral drift.
  * Tilt angle $\theta$ acts mathematically as a **generalized momentum** ($p \sim \theta$), while separation $r$ acts as a **coordinate**.
  * Stokesian equations map to an **effective central-force Hamiltonian**:
    $$\mathcal{H}(r, p) = \frac{p^2}{2m_{\text{eff}}} + \mathcal{V}_{\text{eff}}(r)$$
  * Yields **bound periodic Keplerian orbits** and open hyperbolic scattering in an overdamped viscous fluid!
* **1D Disk Arrays (PRX 2020):** Orientation-drift coupling acts like mass-and-spring chain, supporting acoustic phonon waves and suppressing the Crowley clumping instability!

---

## Slide 14: Taxonomy of Multi-Body Sedimentation Trajectories & Dynamic Animations
| Configuration | Spatial Trajectory $(x, y, z)$ | Hydrodynamic Mechanism | Dynamic Animation |
| :--- | :--- | :--- | :--- |
| **Sphere Pair Drafting** | Parallel vertical descent ($d = \text{const}$) | Mutual Oseen wake drafting | `animations/anim_sphere_pair_sedimentation.gif` |
| **Inclined Ellipsoid (45°)** | Oblique glide with lateral drift | Anisotropic drag tensor ($C_\perp > C_\parallel$) | `animations/anim_ellipsoid_angles_trajectories.gif` |
| **Boomerang Chiral Spin** | 3D corkscrew helical trajectory | Translation-rotation coupling ($B \ne 0$) | `animations/anim_boomerang_chiral_trajectory.gif` |
| **5-Body Linear Chain** | Inverted chevron sagging profile | Additive wake focusing (Crowley instability) | `animations/anim_5body_chain_crowley_instability.gif` |
| **5-Body Ring Cluster** | Symmetrical accelerated descent | Collective suspension wake drafting | Static 4-panel analysis (`*_5body_dynamics.png`) |
| **Pair of Tilted Discs** | Bound periodic Keplerian orbits | Torque-tilt-drift Hamiltonian coupling (Chajwa) | PRL 122, 224501 benchmark replication |

---

## Slide 15: Extrapolation: Machine Learning Surrogate Modeling
* **The Computational Bottleneck:**
  * Multiblob GMRES solves scale as $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$, requiring seconds to minutes per shape.
* **The Collaborative AI Pipeline (Friend's Focus):**
  * **Our Scope (Ground Truth):** High-precision multiblob simulation data across 9 diverse shapes (spheres, cubes, prisms, knots, discs, cylinders, ellipsoids, helices, boomerangs).
  * **Feature Engineering:** 27 invariant geometric descriptors (volume, surface area, sphericity, aspect ratios, moments of inertia, chirality).
  * **ML Surrogate Models:** **Random Forest** and **XGBoost** regressors.
  * **Performance:** Predicts 6-DOF velocity ($V_z$), drift ($V_x$), and drafting factors with **$R^2 > 0.99$** in **sub-milliseconds** ($< 4\,\text{ms}$, $> 500\times$ speedup).
* **Extrapolation to Arbitrary Shapes:** Enables instantaneous sedimentation prediction for novel, irregular 3D micro-debris and biological aggregates without solving PDEs!

---

## Slide 16: Theory-to-Validation Matrix: Connecting Physics to Simulation Data
| Theoretical Principle | Analytical Formulation | Our Numerical Simulation Validation | Proof Location |
| :--- | :--- | :--- | :--- |
| **Vanishing Inertia ($Re \to 0$)** | Instantaneity ($d\mathbf{V}/dt = 0$); linear drag | Multi-scheme comparison matches to $10^{-14}$ precision; $V \propto F$ linearity ($R^2 = 1.000000$) | Slide 7; `schemes_side_by_side_comparison.png` |
| **Universal Normalized Drafting** | Asymptotically recovers $1 + \frac{3}{4}\frac{R}{d}$ as $d/R \gg 1$ | All 9 morphologies collapse onto Stokeslet curve with $< 1\%$ error at $d/R=10$ | Slide 9; `master_normalized_pair_scaling.png` |
| **Boundary Multiblob Convergence** | Discretized RPY blobs approach continuous surface as $N \to \infty$ | Shells converge monotonically to Stokes Law ($98.9\%$ accuracy at $N=2562$) | Slide 6; `figures/*_multires_convergence.png` |
| **The Crowley Instability** | Crowley (1971): wake focusing & lateral in-rush trigger inverted sagging | 5-body chains exhibit $+21.6\% - 37.2\%$ center speedup across 6 shapes | Slide 10; `figures/*_5body_dynamics.png` |
| **Hydrodynamic Rolling & Dipole Scaling** | Wake shear $\nabla \mathbf{u}$ induces torque; decays as $(R/d)^2$ | Counter-rotation confirmed across all pairs; log-log slope $-2.0$ vs $-1.0$ monopole | Slide 11; `multi_body_rotational_dynamics.png` |
| **Exact RPY Drafting Corridors** | Superposition creates low-drag entrainment channel between bodies | Velocity streamplots across 4 geometries show central drafting jet | Slide 12; `flow_fields_single_vs_pair_multishape.png` |
| **Kepler Orbits & Shape Anisotropy** | Chajwa et al. (PRL 2019): Spheres maintain $d(t)=d_0$; discs couple tilt to drift | Sphere pairs maintain $d(t) \equiv d_0$ ($0.0\%$ drift); inclined ellipsoids exhibit strong lateral drift ($V_x \ne 0$) | Slide 8, 13; `lateral_drift_trajectories.png` |
| **Asymmetric Overtaking Drafting** | Orientation mismatch breaks drag symmetry ($\Delta V_z \ne 0$) | Mismatched pairs exhibit $\Delta V_z \approx 12.3$ u/s, driving overtaking | Slide 9; `highres_ellipsoid_angle_pairs.png` |

---

## Slide 17: Conclusions & Master's Defense Takeaways
1. **First-Principles Scaling:** Vanishing Reynolds number ($Re \to 0$) mathematically establishes the linear, instantaneous Stokes regime where Green's functions transform 3D PDEs into discrete boundary solves.
2. **Universal Far-Field Invariance:** All 9 tested particle morphologies asymptotically converge to the continuum point-Stokeslet solution ($1 + \frac{3}{4}\frac{R}{d}$) within $< 1\%$ deviation by $d/R \ge 10.0$.
3. **Monopole vs. Dipole Decays:** Proved that translational drafting decays as $\mathcal{O}(d^{-1})$ (monopole), whereas induced mutual rotation decays as $\mathcal{O}(d^{-2})$ (dipole vorticity gradient).
4. **Finite-Body Crowley Instability:** Proved that 5-body linear arrays across Spheres, Discs, Cylinders, Ellipsoids, Cubes, and Helices exhibit $+21.6\% - 37.2\%$ center speedups, creating physical sagging chevron profiles.
5. **Flow Field Drafting Corridors:** Direct RPY streamplot evaluations reveal the central wake entrainment corridor responsible for $+30\%$ to $+54\%$ pair acceleration.
6. **AI Surrogate Extrapolation:** Multiblob simulation ground truth powers Random Forest/XGBoost regressors ($R^2 > 0.99$), enabling instantaneous ($< 4\,\text{ms}$) drag and drafting predictions for arbitrary particulate shapes.
3. **Morphology-Dependent Near-Field Drafting:** Broadside Discs ($+70.5\%$) and Cylinders ($+60.1\%$) exhibit significantly enhanced near-contact drafting relative to Spheres ($+44.8\%$).
4. **Instabilities & Celestial Analogies:** Linear chains exhibit the classic Crowley sagging instability, while anisotropic discs execute Keplerian bound orbits via orientation-momentum coupling.
5. **Simulation-to-AI Bridge:** Multiblob data serves as the ground-truth engine for Random Forest and XGBoost surrogate modeling of arbitrary 3D particulate shapes.
