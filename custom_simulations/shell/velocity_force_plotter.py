import os
import sys
import argparse
import subprocess

base_dir = r"d:\sedimentation_dynamics"
shell_dir = os.path.join(base_dir, "custom_simulations", "shell")

parser = argparse.ArgumentParser(description="Run velocity vs force simulations and plotting across numerical schemes.")
parser.add_argument("--scheme", choices=["adams_bashforth", "forward_euler", "mobility", "all"], default="all",
                    help="Scheme to execute (adams_bashforth, forward_euler, mobility, or all)")
parser.add_argument("--dt", type=float, default=None,
                    help="Time step dt to run/compare against Mobility (e.g., 0.01)")
parser.add_argument("--rerun", action="store_true", help="Force rerun simulations")
args = parser.parse_args()

if args.dt is not None:
    dt_script = os.path.join(shell_dir, "compare_dt_vs_mobility.py")
    cmd = [sys.executable, dt_script, "--dt", str(args.dt)]
    if args.rerun:
        cmd.append("--rerun")
    print(f"\n{'='*70}\nExecuting dt={args.dt} vs Mobility Comparison:\n{'='*70}")
    subprocess.run(cmd, check=True)
    sys.exit(0)

schemes_to_run = ["adams_bashforth", "forward_euler", "mobility"] if args.scheme == "all" else [args.scheme]

for s in schemes_to_run:
    script_path = os.path.join(shell_dir, s, "velocity_force_plotter.py")
    if os.path.exists(script_path):
        print(f"\n{'='*70}\nExecuting {s.upper()} scheme plotter:\n{'='*70}")
        cmd = [sys.executable, script_path]
        if args.rerun:
            cmd.append("--rerun")
        subprocess.run(cmd, check=True)
    else:
        print(f"Warning: Script not found: {script_path}")

# Run comparison plotter if 'all'
if args.scheme == "all":
    compare_script = os.path.join(shell_dir, "compare_schemes_plotter.py")
    if os.path.exists(compare_script):
        print(f"\n{'='*70}\nGenerating multi-scheme comparison plots:\n{'='*70}")
        subprocess.run([sys.executable, compare_script], check=True)

