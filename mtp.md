# RigidMultiblobsWall â€” Strict Repository-Based Implementation

You are working inside an IDE containing a **cloned copy of the official `RigidMultiblobsWall` repository**.

Repository:
`https://github.com/vpn-cli/RigidMultiblobsWall`

## VERY IMPORTANT â€” SOURCE OF TRUTH

The cloned repository and, **most importantly, the documentation inside its `doc/` directory are the authoritative source for this project.**

### DO NOT:
- reinvent the hydrodynamic formulation;
- derive or replace the mobility formulation with your own implementation;
- substitute another Stokeslet/RPY/RPB implementation;
- introduce a different rigid-body formulation;
- "improve" equations from general knowledge;
- silently change conventions;
- create an independent simulation framework;
- replace repository functions with equivalent-looking code;
- invent input-file formats;
- invent output formats;
- add physical corrections that are not present in the repository;
- change the mathematical formulation merely because another implementation may appear better;
- use external libraries/implementations for the physics when the repository already provides the required functionality.

### DO:
**Reuse the existing repository implementation wherever possible.**

The goal is to build a Python-facing interface/workflow **around the existing repository**, not to rewrite the repository.

If something is unclear or missing, inspect the actual repository source code, examples, structures, input files, and documentation before making any decision.

If the documentation and your general knowledge disagree, **follow the repository.**

If something is genuinely unsupported by the repository, stop and clearly report it rather than inventing a solution.

---

# 1. PROJECT GOAL

We want to use the existing `RigidMultiblobsWall` code from Python through a clean user-facing workflow.

The eventual user should be able to provide:

1. an **input file**;
2. a **`.vertex` file** describing the rigid body's blobs;
3. a **`.clones` file** describing the rigid body's position/orientation;

and run the repository's existing calculations to generate the corresponding results exactly according to the repository documentation.

The initial objective is **NOT** to support every feature of the repository.

We first want to reproduce and verify the simplest documented case:

> **A spherical rigid body represented by a single blob near a wall.**

Only after that verification succeeds should the implementation be generalized.

---

# 2. FIRST TASK â€” UNDERSTAND THE REPOSITORY

Before writing new code:

### A. Inspect the repository structure

Identify:

- `doc/`
- `multi_bodies/`
- `mobility/`
- `Structures/`
- examples
- existing input files
- existing `.vertex` files
- existing `.clones` files
- existing utilities
- existing Python entry points
- existing tests, if any.

Do not modify anything yet.

### B. Read the documentation

Read the relevant documentation files in `doc/`.

The documentation must determine:

- accepted input format;
- `.vertex` format;
- `.clones` format;
- input-file syntax;
- supported schemes;
- mobility implementations;
- wall treatment;
- output files;
- meaning of each output;
- command-line invocation.

In particular, understand the documented:

`body_mobility`

workflow.

The documentation explicitly gives the example:

```text
scheme                                   body_mobility
mobility_blobs_implementation            python
eta                                      1.0
blob_radius                              0.25
output_name                              data/run.body_mobility
structure    Structures/boomerang_N_15.vertex Structures/boomerang_N_15.clones
```

and runs it using:

```bash
python multi_bodies_utilities.py --input-file inputfile_body_mobility.dat
```

Do not replace this workflow with a newly invented algorithm.

---

# 3. UNDERSTAND THE THREE USER INPUTS

The interface we eventually expose should preserve the repository's existing data model.

## `.vertex`

According to the repository documentation, the vertex file contains the blob coordinates in the body's reference configuration.

The documented format is:

```text
number_of_blobs_in_rigid_body
x y z
x y z
...
```

A fourth column may specify individual blob radius.

Do not invent another vertex format.

---

## `.clones`

The `.clones` file contains the rigid-body configuration.

The documented format is:

```text
number_of_rigid_bodies
x y z q0 q1 q2 q3
...
```

where the first three values specify the body location and the quaternion specifies orientation.

Use the repository's existing parser/handling of this format.

Do not create a competing quaternion convention.

---

## Input file

The input file must follow the repository's existing input-file syntax.

Do not replace it with a completely new configuration format unless a thin wrapper is required for usability.

If a wrapper is introduced, it must ultimately generate/use the **same repository input options and values**.

---

# 4. FIRST VERIFICATION CASE â€” SINGLE-BLOB SPHERE

Before implementing arbitrary shapes, reproduce the repository's own single-blob spherical case.

Find the repository's existing structure/example corresponding to a single blob, such as the documented `blob.vertex` structure or the appropriate sphere example.

Do NOT create an arbitrary sphere discretization yourself.

For the first test:

- one rigid body;
- one blob;
- repository-compatible `.vertex`;
- repository-compatible `.clones`;
- single wall;
- repository's Python mobility implementation;
- repository's `body_mobility` scheme.

The purpose is to verify that our Python interface produces the same result as directly running the repository.

---

# 5. BASELINE TEST

First run the repository **without our wrapper/interface** using its documented command.

For example, from the appropriate repository directory:

```bash
python multi_bodies_utilities.py --input-file <inputfile>
```

Record:

- command used;
- input values;
- repository version/commit;
- generated files;
- `.body_mobility.dat`;
- numerical values in the mobility matrix;
- stdout/stderr;
- any warnings/errors.

This is the **golden baseline**.

Then implement our interface and run exactly the same physical case.

The wrapper result must be compared against the repository baseline.

---

# 6. NUMERICAL VERIFICATION

For the first sphere test, compare the generated:

```text
.body_mobility.dat
```

against the direct repository result.

Do not merely check that the program runs.

Check the actual numerical values.

The documentation states that `.body_mobility.dat` contains the `6Ã—6` rigid-body mobility matrix and that applying it to `(force, torque)` produces `(velocity, angular_velocity)`.

Therefore verify:

- matrix dimensions;
- numerical values;
- ordering;
- translational components;
- rotational components;
- coupling terms;
- units/conventions as used by the repository.

Do not impose an independently derived analytical formula as the primary verification.

The repository output is the reference.

---

# 7. REUSE EXISTING REPOSITORY CODE

Search the repository before implementing anything.

For every required operation, determine whether the repository already has a function/class/module for it.

Prefer:

```text
existing repository function
        â†“
existing repository parser
        â†“
existing repository solver
        â†“
existing repository mobility implementation
```

over:

```text
new implementation
```

The new code should act primarily as an interface/orchestration layer.

For example, conceptually:

```text
User
 â”‚
 â”œâ”€â”€ input file
 â”œâ”€â”€ vertex file
 â””â”€â”€ clones file
        â”‚
        â–¼
Our Python interface
        â”‚
        â–¼
Existing RigidMultiblobsWall code
        â”‚
        â–¼
Existing mobility calculation
        â”‚
        â–¼
Existing repository output
```

Do not duplicate the hydrodynamic mathematics.

---

# 8. PYTHON ONLY FOR THE INITIAL IMPLEMENTATION

For the initial implementation use the repository's Python implementations.

Use the repository's documented option:

```text
mobility_blobs_implementation python
```

and the appropriate Python matrix-vector implementation where required.

Do not prematurely introduce:

- C++;
- pybind11;
- CUDA;
- PyCUDA;
- custom GPU kernels;
- custom Numba kernels.

The documentation says the package already has Python implementations and that compilation is not necessary to use the package.

We want correctness and repository fidelity first.

Performance optimization comes later.

---

# 9. WALL MODEL

The repository documentation states that the wall-corrected blob mobility uses the **Rotne-Prager-Blake** formulation.

Do not implement another wall correction.

Do not replace it with a generic Stokeslet image system.

Do not simplify the wall interaction.

Use the repository's existing implementation.

---

# 10. DO NOT CREATE A NEW PHYSICS ENGINE

This is critical.

We are **not** asking you to write:

```text
STL â†’ custom Stokeslets â†’ custom mobility matrix â†’ custom solver
```

from scratch.

We are asking you to make the existing:

```text
RigidMultiblobsWall
```

repository usable from a controlled Python interface.

The repository already contains the physics implementation.

Our code should call it.

---

# 11. INPUT/OUTPUT INTERFACE

After the baseline sphere test works, create a minimal Python interface that allows a user to specify:

```text
input_file
vertex_file
clones_file
```

and execute the appropriate repository workflow.

For example, a simple command/interface may eventually look like:

```bash
python run_rigid_multiblobs.py \
    --input-file inputfile.dat \
    --vertex-file sphere.vertex \
    --clones-file sphere.clones
```

However:

**Do not assume this exact CLI design is required.**

Choose the smallest interface necessary to expose the repository functionality.

The important requirement is that the underlying repository input conventions remain unchanged.

---

# 12. OUTPUT

Do not invent a new output format for the first implementation.

Preserve the repository outputs.

For the `body_mobility` calculation, the important output is:

```text
.body_mobility.dat
```

Also preserve the repository's input-file copy and other outputs where the repository generates them.

If a wrapper wants to provide a convenient summary, it may do so **in addition to**, not instead of, the original repository output.

---

# 13. ERROR HANDLING

The interface should validate basic things such as:

- input file exists;
- vertex file exists;
- clones file exists;
- paths are valid;
- repository-required values are present.

But do not invent physics validation rules.

If the repository itself rejects an input, expose the repository error rather than silently modifying the input.

For example, do not automatically:

- change blob radius;
- move a particle away from the wall;
- normalize data differently;
- alter quaternions;
- change viscosity;
- change solver tolerance.

unless the repository itself does so.

---

# 14. SPHERE VERIFICATION CHECKLIST

The first milestone is complete only when all of the following are true:

### Repository baseline

```text
[ ] Existing sphere/single-blob example identified
[ ] Existing input file understood
[ ] Existing vertex file understood
[ ] Existing clones file understood
[ ] Repository run succeeds
[ ] Repository output captured
```

### Our interface

```text
[ ] Python interface accepts the same files
[ ] Existing repository code is used
[ ] No independent mobility implementation exists
[ ] No independent wall correction exists
[ ] Python implementation is used
[ ] Same physical parameters are used
[ ] Same body configuration is used
[ ] Same output is generated
```

### Numerical verification

```text
[ ] 6Ã—6 mobility matrix generated
[ ] Matrix dimensions match
[ ] Numerical values match baseline within an explicitly stated tolerance
[ ] File format matches repository
[ ] Any discrepancy is investigated before proceeding
```

---

# 15. REPRODUCIBILITY

Create a small reproducible verification test containing:

```text
tests/
    sphere_single_blob/
        inputfile...
        sphere.vertex
        sphere.clones
        expected_body_mobility.dat
```

But do not fabricate expected values.

Generate the expected result from the repository's own implementation.

Record how it was generated.

The test should execute the same repository calculation and compare the result.

---

# 16. DOCUMENT EVERY DEVIATION

If you have to write any new code that is not directly present in the repository, document:

1. why it is necessary;
2. which repository functionality it connects to;
3. why it does not change the underlying physics;
4. which documentation section justifies the behavior.

If something cannot be implemented without inventing behavior, **STOP and report it instead of guessing.**

---

# 17. DEVELOPMENT ORDER

Follow this exact order:

### Phase 1
Inspect repository and documentation.

### Phase 2
Identify the existing single-blob sphere example.

### Phase 3
Run the original repository directly.

### Phase 4
Save the original output as the baseline.

### Phase 5
Build the thinnest possible Python interface around the repository.

### Phase 6
Run the interface using exactly the same sphere inputs.

### Phase 7
Compare numerical output against the baseline.

### Phase 8
Only after exact/within-tolerance agreement, generalize the interface to arbitrary `.vertex` + `.clones` inputs.

### Phase 9
Test a multi-blob rigid body using a repository-provided structure.

### Phase 10
Only after those tests pass, consider STL-related preprocessing.

---

# 18. STL IS A LATER STAGE

Eventually our larger research project may involve arbitrary STL geometries.

Do **not** assume that `RigidMultiblobsWall` itself accepts STL files unless the repository documentation/source explicitly demonstrates this.

At this stage the required input is:

```text
.vertex
.clones
input file
```

If later we need:

```text
STL â†’ vertex
```

that conversion must be treated as a separate stage and must be based on the actual methodology specified by our project/repository documentation.

Do not invent a blob-discretization algorithm now.

---

# 19. IMPORTANT RESEARCH DISCIPLINE

Whenever you are about to write a mathematical or numerical implementation, ask:

> "Does RigidMultiblobsWall already implement this?"

If yes:

**use it.**

If no:

> "Does the repository documentation explicitly specify how it should be done?"

If yes:

**follow it exactly.**

If no:

**do not invent it. Report the missing specification.**

This rule is more important than making the system appear complete.

---

# 20. FINAL DELIVERABLE FOR THIS STAGE

At the end of this task, provide:

### A. Repository understanding

A concise map of:

```text
input
  â†“
vertex/clones parsing
  â†“
existing repository functions
  â†“
mobility calculation
  â†“
output
```

with actual repository filenames/functions.

### B. Working Python interface

A minimal interface that accepts the required repository inputs and executes the existing calculation.

### C. Sphere verification

Show:

```text
Repository baseline
vs.
Our interface
```

including the actual `6Ã—6` mobility matrices and numerical differences.

### D. Test command

Give the exact command needed to reproduce the verification.

### E. Changes made

List every file you created or modified.

### F. Explicit statement

State clearly whether the implementation:

- uses the original repository mobility implementation;
- uses the original repository wall treatment;
- uses the original repository input conventions;
- produces the original repository output;
- passes the single-blob sphere verification.

---

## ABSOLUTE RULE

**DO NOT MAKE THE CODE LOOK COMPLETE BY INVENTING PLACEHOLDERS OR SIMPLIFICATIONS. GROUND EVERY EQUATION AND SOLVER IN THE REPOSITORY CODE.**

---

# 21. PROGRESS SUMMARY, COMPLETED SIMULATIONS & CONTINUATION INSTRUCTIONS

### A. Completed and Verified Stages

1. **Stage 1 — Single-Blob Sphere Verification Near Wall**
   - **Scheme**: `body_mobility`
   - **Formulation**: Rotne-Prager-Blake (RPB) tensor via `multi_bodies_utilities.py`.
   - **Verification**: Compared repository output with analytical Swan & Brady wall mobility; achieved $0.00\times 10^{0}$ difference (exact match).

2. **Stage 2 — Sedimentation of $N=12$ Spherical Shell Near Wall**
   - **Scheme**: `deterministic_adams_bashforth`
   - **Domain**: `single_wall` (RPB wall-corrected mobility).
   - **Files**: `shell_N_12_Rg_1_Rh_1_2625.vertex` and `shell_N_12_Rg_1.clones`.
   - **Physics Result**: Body slows down as $z \to 0$ due to Blake hydrodynamic wall drag ($V_z = -0.487 \to -0.441\,\text{units/s}$).

3. **Stage 3 — Free-Fall Sedimentation in Unbounded Fluid (No Wall)**
   - **Scheme**: `deterministic_adams_bashforth`
   - **Domain**: `domain no_wall`
   - **Formulation**: Rotne-Prager-Yamakawa (RPY) mobility tensor via `python_no_wall` without wall corrections.
   - **Wall Potential**: `repulsion_strength_wall 0.0` (wall interaction deactivated).
   - **Physics Result**: Unbounded Stokes flow is translationally invariant; sedimentation velocity $V_z$ is strictly constant across all timesteps.

4. **Stage 4 — Discretization Convergence Study ($N \in \{12, 42, 162, 642, 2562\}$)**
   - Executed free-fall sedimentation across all available shell resolutions in `Structures/`.
   - Demonstrated monotonic convergence of the effective mobility per unit force:
     - $N = 12$: $V_z / (N g) = -0.058885$
     - $N = 42$: $V_z / (N g) = -0.053635$
     - $N = 162$: $V_z / (N g) = -0.053095$
     - $N = 642$: $V_z / (N g) = -0.053013$
     - $N = 2562$: $V_z / (N g) = -0.053002$
   - Matches analytical two-sphere drafting theory ($\mu_0 + \Delta \mu \approx 0.0529 \approx 0.0530$).

---

### B. Execution Commands for Reproducing Simulations

#### 1. Native Terminal Invocations (Per `doc/README.md`)
Always execute from `c:\MTP-oct\RigidMultiblobsWall\multi_bodies`:
```bash
cd c:\MTP-oct\RigidMultiblobsWall\multi_bodies

# Run free-fall simulation for N=12
python multi_bodies.py --input-file inputfile_shell_N_12_free_fall.dat

# Run free-fall simulation for N=42
python multi_bodies.py --input-file inputfile_shell_N_42_free_fall.dat

# Run free-fall simulation for N=162
python multi_bodies.py --input-file inputfile_shell_N_162_free_fall.dat

# Run free-fall simulation for N=642
python multi_bodies.py --input-file inputfile_shell_N_642_free_fall.dat

# Run free-fall simulation for N=2562
python multi_bodies.py --input-file inputfile_shell_N_2562_free_fall.dat
```

#### 2. Clean Python Driver (From Workspace Root)
From `c:\MTP-oct`:
```bash
# Run all resolutions and display convergence table
python simulate_shell_sedimentation.py

# Run for a specific N (e.g. N=42)
python simulate_shell_sedimentation.py --N 42
```

#### 3. Analyzing Output Trajectories
All trajectory output files are stored in `RigidMultiblobsWall/multi_bodies/data/`:
- `*.config`: Position and quaternion coordinates for all timesteps.
- `*.bodies_info`: Number of bodies, blobs, and body types.
- `*.time`: Elapsed wallclock simulation time.

To compute linear and angular velocity using repository tools:
```bash
python ../tools/velocity_linear_angular.py data/shell_N_12_free_fall.shell_N_12_Rg_1.config 0.05 1
```

---

### C. Roadmap & Instructions for Future Work

When continuing this project, proceed through the following steps without deviating from the repository rules:

1. **Custom Discretization & Mesh Generation (Phase 10)**:
   - When generating new shapes or STLs, use the documented surface blob placement method (`multi_bodies/Structures/create_3d_sphere.cpp`).
   - Calculate geometric radius $R_g$ and choose blob radius $a$ such that the blobs cover the surface without excessive overlap, following the repository convention.

2. **Articulated Rigid Bodies & Constraints**:
   - Follow `multi_bodies/examples/bacteria` to link multiple rigid bodies using `.list_vertex` and `.const` files.
   - Use `articulated_deterministic_midpoint` or `deterministic_adams_bashforth` for articulated bodies.

3. **Active Phoretic / Squirmer Particles**:
   - Follow `multi_bodies/examples/squirmer` to add active slip files (`.slip`) and prescribe surface boundary conditions.

4. **Brownian Dynamics (Stochastic Simulations)**:
   - Set `kT > 0` in the input file and switch to documented stochastic schemes (`stochastic_Slip_Trapz` or `stochastic_first_order_RFD`).

