#!/usr/bin/env python3
import os
import subprocess
import sys

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def main():
    print("--- [Pre-Commit Hook] Checking for untracked memory/plan files ---")
    
    # 1. Get untracked files
    untracked = run_command(["git", "ls-files", "--others", "--exclude-standard"])
    if not untracked:
        untracked_files = []
    else:
        untracked_files = untracked.split("\n")

    # 2. Identify files in relevant directories
    auto_stage_dirs = ["memory/", "docs/", "scratch/"]
    to_stage = [f for f in untracked_files if any(f.startswith(d) for d in auto_stage_dirs)]

    if to_stage:
        print(f"Found {len(to_stage)} untracked file(s) in memory/docs/scratch. Auto-staging:")
        for f in to_stage:
            print(f"  + {f}")
            run_command(["git", "add", f])
    
    # 3. Create the memory update pending flag (per CLAUDE.md protocol)
    flag_dir = ".claude"
    flag_file = os.path.join(flag_dir, ".memory_update_pending")
    
    if not os.path.exists(flag_dir):
        os.makedirs(flag_dir)
        
    with open(flag_file, "w") as f:
        f.write("Update required after commit on " + os.popen("date /T").read().strip() if os.name == 'nt' else os.popen("date").read().strip())
    
    # Note: We don't stage the flag file itself, it stays local to remind the next session.
    print(f"Flagged memory update at {flag_file}")
    print("--- [Pre-Commit Hook] Complete ---")

if __name__ == "__main__":
    main()
