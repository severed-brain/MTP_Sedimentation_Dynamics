# AI Agent Automation Knowledge Base
**Target System:** `RigidMultiblobsWall` (Stokesian Dynamics for Sedimentation)
**Goal:** Provide an AI agent with all necessary context, commands, mathematical formulas, and code snippets to automate simulations for **any arbitrary shape** in the future.

---

## 1. Core Architecture & Execution
The underlying physics engine is written in Fortran/C++/CUDA, with a Python wrapper located at `RigidMultiblobsWall/multi_bodies/multi_bodies.py`.

### Execution Requirements
To execute a simulation from anywhere in the file system, the AI **must** inject the `PYTHONPATH` so the compiled libraries load:
```python
import os
import subprocess

env = os.environ.copy()
env["PYTHONPATH"] = r"d:\sedimentation_dynamics\RigidMultiblobsWall\multi_bodies;d:\sedimentation_dynamics\RigidMultiblobsWall"

cmd = ["python", r"d:\sedimentation_dynamics\RigidMultiblobsWall\multi_bodies\multi_bodies.py", "--input-file", "my_input.dat"]
subprocess.run(cmd, env=env, cwd="path/to/my/folder")
```
*Rule of Thumb:* Always run the subprocess with `cwd` set to the folder where the `.dat`, `.vertex`, and `.clones` files physically reside.

---

## 2. Anatomy of the Required Files

To simulate an arbitrary shape, the AI needs to generate 3 files:

### A. The Geometry File (`.vertex`)
This file defines the surface mesh of the shape using interacting blobs.
**Format:**
```text
<N> <Average_Blob_Distance>
<X1> <Y1> <Z1>
<X2> <Y2> <Z2>
...
```
*   **N:** Number of blobs.
*   **Average Blob Distance:** The average distance between neighboring points. (Important: The optimal `blob_radius` parameter in the `.dat` file is typically $\sim 0.5 \times$ this distance to prevent fluid leaking through the shell).
*   **Coordinates:** Must be relative to the Center of Mass $(0,0,0)$.
*   **Tip for AI:** To create new shapes quickly without CAD, you can apply Affine Transformations (stretching/compressing) directly to a baseline spherical `.vertex` file.

### B. The Orientation File (`.clones`)
This file places the geometry into the domain and applies 3D rotations.
**Format:**
```text
<Number_of_Bodies>
<X_pos> <Y_pos> <Z_pos> <qw> <qx> <qy> <qz>
```
*   **Number of Bodies:** Usually `1` for single-body sedimentation.
*   **Position:** Usually `0 0 0` to start at the origin.
*   **Quaternion ($qw, qx, qy, qz$):** Controls the release angle.
    *   *Example:* To tilt by angle $\alpha$ around the Y-axis: $qw = \cos(\alpha/2)$, $qx=0$, $qy=\sin(\alpha/2)$, $qz=0$.

### C. The Configuration File (`.dat`)
This configures the fluid and solver parameters.
**Crucial Parameters for AI Automation:**
*   `n_steps`: Number of time steps. (e.g., 20 for basic spheres, 100 for long angle drifts).
*   `dt`: Time step size (default 0.05).
*   `g`: Force per blob. **Formula: $g = \text{Total Force} / N$**
*   `blob_radius`: Hydrodynamic radius of the individual blobs. **Formula: $a \approx 0.5 \times \sqrt{\text{Total Surface Area} / N}$**.
*   `output_name`: Where to save the output trajectory (e.g., `data/my_shape_output`). *Note: The AI must ensure the `data/` folder exists before running!*
*   `structure`: `<vertex_file_name> <clones_file_name>` (Must point to the exact names of the files generated in steps A and B).

---

## 3. Data Extraction (Parsing the Output)
The solver outputs a `.config` file tracking the XYZ coordinates of every blob at every time step. 
**How the AI should extract Terminal Velocity ($V_z$) and Lateral Drift ($V_x$):**
1. Open `<output_name>.<clones_prefix>.config`.
2. The file is separated into "frames" by a line containing the number of bodies (e.g., `1`).
3. The lines following are the XYZ coordinates of the Center of Mass.
4. Read the $Z$-coordinate of the first frame ($Z_0$) and the last frame ($Z_{end}$).
5. Calculate $V_z = |(Z_{end} - Z_0) / (n\_steps \times dt)|$.

---

## 4. Universal Physical Concepts for AI Reasoning
If the AI is tasked with analyzing data or answering questions, it must use these principles:
*   **Stokes' Law Linearity:** In creeping flow, Terminal Velocity scales strictly linearly with Force.
*   **Hydrodynamic vs Geometric Radius:** Geometries are often mathematically shrunk by a slight factor (e.g., $1.0 / 1.0530 \approx 0.95$) so that their effective fluid "Hydrodynamic Radius" ($R_h$) perfectly equals $1.0$ for comparison against analytical equations.
*   **Orientation Drag:** Shapes falling "End-On" (parallel to gravity, smallest cross-section) fall faster. Shapes falling "Broad-Side" (perpendicular, largest cross-section) fall slower.
*   **Lateral Drift:** Symmetric anisotropic objects (like ellipsoids) falling at an angle (e.g., $45^\circ$) will experience a horizontal drift component. This drift is zero at $0^\circ$ and $90^\circ$.
*   **Normalization:** When plotting, Terminal Velocity $|U|$ is often normalized by dividing it by the terminal velocity of a perfect sphere of equal volume. This creates a dimensionless, universal metric.
