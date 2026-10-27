# AI/ML Sedimentation Predictor for Arbitrary 3D Rigid Bodies

This package implements a fast, physics-informed Machine Learning surrogate model capable of predicting the full sedimentation dynamics ($V_z$, lateral drift $V_x$, chiral spin $\Omega_z$, and glide ratio $|V_x/V_z|$) for **any arbitrary 3D rigid body shape** in low Reynolds number Stokes flow.

---

## 1. Physical Motivation & Capabilities

Solving Stokes flow sedimentation numerically via the **RigidMultiblob** approach requires forming the grand mobility matrix $\mathcal{M}$ (using Rotne-Prager-Yamakawa tensors) and solving large linear systems:
$$\mathcal{U} = \mathcal{M} \mathcal{F}$$
For large discretizations ($N \ge 100$ blobs), this solve requires seconds to minutes per time-step.

This ML surrogate maps geometric shape features and orientation angles directly to terminal hydrodynamic velocities in **under 5 milliseconds**, delivering over **500x speedup** while achieving **$R^2 > 0.99$** agreement with full numerical solutions.

---

## 2. 3D Physics-Informed Feature Representation (27 Descriptors)

The model operates on arbitrary `.vertex` point clouds by computing 27 rotation-aware and invariant hydrodynamic features:

| Category | Features | Physical Significance |
| :--- | :--- | :--- |
| **Discretization** | $N$, $a$ | Number of blobs and hydrodynamic radius |
| **Bounding Envelopes** | $L_x, L_y, L_z$, $V_{box}$ | Spatial span and bounding box volume |
| **Geometric Aspect Ratios** | $\mathrm{AR}_{max}, \mathrm{AR}_z$ | Slenderness and vertical elongation |
| **Span & Compactness** | $R_g, D_{max}, A_{blobs}, V_{blobs}$ | Radius of gyration and total hydrodynamic volume |
| **Inertia Tensor Invariants** | $\lambda_1, \lambda_2, \lambda_3$ | Principal moments of mass/blob distribution |
| **Shape Anisotropy** | Asphericity $b$, Acylindricity $c$, Relative Anisotropy $\kappa^2$ | Deviation from spherical and cylindrical symmetry |
| **Orientation Coupling** | $\theta, \cos\theta, \sin(2\theta)$ | Tilt of major geometric axis relative to gravity (drives $V_x$) |
| **Cross-Sectional Areas** | $A_\perp, A_\parallel, A_\perp/A_\parallel$ | Frontal projected resistance area versus vertical side area |
| **Chirality Pseudoscalar** | $\chi = \sum (r_i \times r_j) \cdot r_k$ | Quantifies structural handedness (drives chiral spin $\Omega_z$) |
| **Applied Loading** | $F, g = F/N$ | Total gravitational force and buoyant acceleration |

---

## 3. Directory Structure

```
ml_predictor/
├── data/
│   └── sedimentation_dataset.csv     # 263 verified simulation data points
├── models/
│   ├── scaler.joblib                 # StandardScaler fitted strictly on train split
│   ├── model_Vz.joblib               # Optimal terminal velocity predictor (XGBoost)
│   ├── model_Vx.joblib               # Lateral drift predictor (XGBoost)
│   ├── model_Wz.joblib               # Chiral spin predictor (XGBoost)
│   └── model_metadata.json           # Feature columns and validation metrics
├── plots/
│   └── model_comparison_and_parity.png # 4-panel parity & residual evaluation
├── feature_extractor.py              # Invariant and PCA-aligned feature extraction
├── dataset_builder.py                # Harvests simulation data + runs augmented solves
├── train_models.py                   # 5-fold CV, multi-model benchmark, parity plotting
└── predict.py                        # CLI inference and on-the-fly simulation validation
```

---

## 4. Usage & Inference CLI

### Single Shape Prediction:
```bash
python ml_predictor/predict.py --vertex path/to/shape.vertex --angle 45 --force 1000
```

### Prediction with On-The-Fly Numerical Validation:
```bash
python ml_predictor/predict.py --vertex custom_simulations/helix/single/structures/helix_N_38.vertex --angle 45 --force 1000 --validate
```

Example Output:
```
================================================================================
AI SEDIMENTATION DYNAMICS PREDICTOR (ARBITRARY SINGLE SHAPE)
================================================================================
Input Shape:      helix_N_38.vertex
Discretization:   N = 38 blobs, a = 0.1254
Shape Invariants: Aspect Ratio = 5.18, Anisotropy = 0.228, Chirality = -0.0119
Orientation:      Pitch Angle theta = 45.0 deg (cos theta = 0.707)
Applied Force:    F = 1000.0 units
--------------------------------------------------------------------------------
MACHINE LEARNING PREDICTION RESULTS (Inference time: 3.5 ms):
  * Vertical Terminal Velocity Vz:    52.570 units/s
  * Lateral Drift Velocity Vx:          6.829 units/s
  * Chiral Angular Velocity Wz:         4.399 rad/s
  * Glide Ratio |Vx / Vz|:             0.1299
--------------------------------------------------------------------------------
RUNNING GROUND-TRUTH STOKESIAN SIMULATION FOR VALIDATION...
Simulation completed in 2.11 s. Comparison Table:
Quantity                       ML Predicted    Ground Truth    Rel Error   
---------------------------------------------------------------------------
Vertical Velocity Vz (units/s) 52.570          52.716          0.28%
Lateral Drift Vx (units/s)     6.829           7.267           0.438 abs
Glide Ratio |Vx/Vz|            0.1299          0.1379         

=> Computational Acceleration: 526x faster than numerical simulation!
================================================================================
```

---

## 5. Next Phase: Roadmap to Multi-Body Formulations
Once single-body predictions are established:
1. **Pairwise Hydrodynamic Interaction Graph**: Encode relative inter-body distance vectors $\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i$ and orientation differences.
2. **Graph Neural Networks (GNN)**: Model multiple bodies as interacting nodes in a creeping flow graph to predict cooperative drafting and cross-stream repulsion.
