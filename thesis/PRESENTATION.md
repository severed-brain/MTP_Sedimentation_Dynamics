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

## Slide 5: Multi-Resolution Shell Discretization & Convergence
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

## Slide 6: Computational Complexity & Scheme Evaluation
* Execution wall-clock time for 5 integration steps:
  * $N=12 \to 0.07$ s
  * $N=162 \to 0.52$ s
  * $N=642 \to 3.79$ s
  * $N=2562 \to 63.76$ s
* **Scheme Comparison (Adams-Bashforth vs. Forward Euler vs. Mobility):**
  * Identical terminal velocity to $10^{-14}$ precision because acceleration is zero ($d\mathbf{V}/dt = 0$).
  * **Direct Mobility Scheme** requires only **1 solve**, making it $10\times - 20\times$ faster for finding terminal velocity and mobility matrices!

---

## Slide 7: Non-Spherical Particles: Prolate vs. Oblate Ellipsoids
* Shape anisotropy modulates terminal drag and settling speed:
  * **Prolate ($1 \times 1 \times 2$):** Slender profile along settling axis $\implies$ Reduced drag, faster sedimentation ($V_0 = 40.45$).
  * **Oblate ($1 \times 2 \times 2$):** Broad frontal area $\implies$ Elevated drag, slower sedimentation ($V_0 = 33.10$).
* **Orientation Coupling & Lateral Drift:**
  * Inclined ellipsoids ($\theta \in [0^\circ, 90^\circ]$) exhibit strong off-diagonal mobility coupling ($M_{xz} \ne 0$), reaching peak lateral drift at $\theta = 45^\circ$.
  * Validated against benchmark literature (*Phys. Rev. E 109, 065302, 2024*).

---

## Slide 8: Multi-Body Pair Hydrodynamic Drafting
* Pair drafting scaling across separation ratios $d/R \in [2.0, 12.0]$:
  * **Spherical Shells ($N = 42$):** $V_z / V_0 = 1.4480$ at $d/R = 2.0$, decaying to $1.0705$ at $d/R = 12.0$.
  * **Circular Discs ($N = 37$, Broadside):** $V_z / V_0 = 1.7053$ ($+70.5\%$ speedup), closely tracking broadside Stokeslet $1 + \frac{2}{\pi}\frac{R}{d}$.
  * **Cylinders ($N = 86$, Side-by-Side):** $V_z / V_0 = 1.6014$ ($+60.1\%$ speedup).
  * **Prolate Ellipsoids ($N = 42$, Major Axis):** $V_z / V_0 = 1.3608$ ($+36.1\%$ speedup; exact match with Stokeslet at $d/R = 6.0$).
* **Universal Far-Field Invariance:** All geometries collapse onto the point-Stokeslet continuum decay ($1 + \frac{3}{4}\frac{R}{d}$) within $< 3.5\%$ error by $d/R \ge 10.0$.

---

## Slide 9: Collective Sedimentation & The Crowley Instability
* **5-Body Linear Horizontal Array ($M = 5$):**
  * Symmetrical flanks focus downward momentum at the array center.
  * **Center-to-Edge Velocity Gradients:**
    * Spheres: $+27.5\%$ at $d/R = 3.0$
    * Discs: $+36.2\%$ at $d/R = 3.0$
    * Cylinders: $+37.2\%$ at $d/R = 3.0$
    * Ellipsoids: $+24.8\%$ at $d/R = 4.0$
  * **Crowley Instability Connection (Crowley 1971):**
    * Mutual drafting accelerates center bodies $\implies$ lateral fluid convergence draws particles closer $\implies$ runaway sagging into inverted-U / chevron profiles, triggering droplet pinch-off.
* **5-Body Pentagonal Ring Cluster ($M = 5$):**
  * Symmetrical collective drafting accelerates entire suspension:
    * Discs: $V_z / V_0 = 2.4968$ ($+149.7\%$ speedup at $d/R = 2.0$)
    * Spheres: $V_z / V_0 = 2.4403$ ($+144.0\%$ speedup)
    * Ellipsoids: $V_z / V_0 = 2.3479$ ($+134.8\%$ speedup)

---

## Slide 10: Non-Spherical Interactions & Keplerian Orbits (Rahul Chajwa's Work)
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

## Slide 11: Taxonomy of Multi-Body Sedimentation Trajectories
| Configuration | Spatial Trajectory $(x, y, z)$ | Hydrodynamic Mechanism |
| :--- | :--- | :--- |
| **Single Isolated Sphere** | Purely vertical linear fall | Isotropic Stokes drag ($F = 6\pi\eta R V_0$) |
| **Horizontal Sphere Pair** | Parallel vertical descent ($d = \text{const}$) | Mutual Oseen wake drafting |
| **Vertical Tandem Pair** | Trailing capture / leap-frogging | Asymmetric drafting in leading wake |
| **Pair of Tilted Discs** | Bound periodic Keplerian orbits / scattering | Torque-tilt-drift Hamiltonian coupling (Chajwa) |
| **5-Body Linear Chain** | Inverted chevron sagging profile | Additive wake focusing (Crowley instability) |
| **5-Body Ring Cluster** | Symmetrical accelerated descent | Collective suspension wake drafting |
| **Inclined Ellipsoid** | Oblique trajectory with lateral drift | Anisotropic drag tensor ($C_\perp > C_\parallel$) |

---

## Slide 12: Extrapolation: Machine Learning Surrogate Modeling
* **The Computational Bottleneck:**
  * Multiblob GMRES solves scale as $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$, requiring seconds to minutes per shape.
* **The Collaborative AI Pipeline (Friend's Focus):**
  * **Our Scope (Ground Truth):** High-precision multiblob simulation data across diverse shapes (spheres, cubes, prisms, knots, discs, cylinders, ellipsoids).
  * **Feature Engineering:** 27 invariant geometric descriptors (volume, surface area, sphericity, aspect ratios, moments of inertia, chirality).
  * **ML Surrogate Models:** **Random Forest** and **XGBoost** regressors.
  * **Performance:** Predicts 6-DOF velocity ($V_z$), drift ($V_x$), and drafting factors with **$R^2 > 0.99$** in **sub-milliseconds** ($< 4\,\text{ms}$, $> 500\times$ speedup).
* **Extrapolation to Arbitrary Shapes:** Enables instantaneous sedimentation prediction for novel, irregular 3D micro-debris and biological aggregates without solving PDEs!

---

## Slide 13: Theory-to-Validation Matrix: Connecting Physics to Simulation Data
| Theoretical Principle | Analytical Formulation | Our Numerical Simulation Validation | Proof Location |
| :--- | :--- | :--- | :--- |
| **Vanishing Inertia ($Re \to 0$)** | Instantaneity ($d\mathbf{V}/dt = 0$); linear drag | Multi-scheme comparison matches to $10^{-14}$ precision; $V \propto F$ linearity ($R^2 = 1.000000$) | Slide 6; `schemes_side_by_side_comparison.png` |
| **Far-Field Point Stokeslet** | Green's function Oseen tensor: $V_z / V_0 \approx 1 + \frac{3}{4}\frac{R}{d}$ | All 7 geometries collapse onto Stokeslet curve ($< 3.5\%$ error at $d/R = 10 - 12$) | Slide 8; `figures/*_pair_trajectories.png` |
| **Boundary Multiblob Convergence** | Discretized RPY blobs approach continuous surface as $N \to \infty$ | Shells converge monotonically to Stokes Law ($98.9\%$ accuracy at $N=2562$) | Slide 5; `figures/*_multires_convergence.png` |
| **The Crowley Instability** | Crowley (1971): wake focusing & lateral in-rush trigger inverted sagging | 5-body chains exhibit $+24.8\% - 37.2\%$ center speedup, creating physical sagging chevrons | Slide 9; `figures/*_5body_dynamics.png` |
| **Kepler Orbits & Shape Anisotropy** | Chajwa et al. (PRL 2019): Spheres maintain $d(t)=d_0$; discs couple tilt to drift | Sphere pairs maintain $d(t) \equiv d_0$ ($0.0\%$ drift); inclined ellipsoids exhibit strong lateral drift ($V_x \ne 0$) | Slide 7, 10; `lateral_drift_trajectories.png` |
| **Planar Wake Amplification** | Discs displace larger lateral fluid envelope than spheres | At $d/R = 2.0$, disc pair drafting reaches $+70.5\%$ ($V_z/V_0 = 1.7053$) vs. $+44.8\%$ for spheres | Slide 8; `disc_pair_trajectories.png` |

---

## Slide 14: Conclusions & Master's Defense Takeaways
1. **First-Principles Scaling:** Vanishing Reynolds number ($Re \to 0$) mathematically establishes the linear, instantaneous Stokes regime where Green's functions transform 3D PDEs into discrete boundary solves.
2. **Universal Far-Field Invariance:** All 7 tested particle morphologies asymptotically converge to the continuum point-Stokeslet solution ($1 + \frac{3}{4}\frac{R}{d}$) within $< 3.5\%$ deviation by $d/R \ge 10.0$.
3. **Morphology-Dependent Near-Field Drafting:** Broadside Discs ($+70.5\%$) and Cylinders ($+60.1\%$) exhibit significantly enhanced near-contact drafting relative to Spheres ($+44.8\%$).
4. **Instabilities & Celestial Analogies:** Linear chains exhibit the classic Crowley sagging instability, while anisotropic discs execute Keplerian bound orbits via orientation-momentum coupling.
5. **Simulation-to-AI Bridge:** Multiblob data serves as the ground-truth engine for Random Forest and XGBoost surrogate modeling of arbitrary 3D particulate shapes.
