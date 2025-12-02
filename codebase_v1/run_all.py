import numpy as np
import os
import glob
import sys
import subprocess
from scipy.spatial import ConvexHull

# --- 1. Configuration ---

# Commands for all your C++ algorithms
ALGO_COMMANDS = {
    "Monotone (Base)": ["./monotone_chain"],
    "Monotone (Optimized)": ["./optimized_monotone_chain"],
    "QuickHull (C++)": ["./quickhull"],
    "Graham Scan (C++)": ["./graham_scan"]
}

# Source and destination folders
DATASET_DIR = "datasets"
SCIPY_OUTPUT_DIR = "SCIPY_OUTPUT"


# --- 2. SciPy Ground Truth Generation ---

def run_scipy_generation(dataset_paths):
    """
    Runs the SciPy analysis on all datasets and saves the
    results to the SCIPY_OUTPUT_DIR.
    """
    print(f"--- 1. Generating SciPy Ground Truth in '{SCIPY_OUTPUT_DIR}' ---")
    os.makedirs(SCIPY_OUTPUT_DIR, exist_ok=True)

    for dataset_path in dataset_paths:
        base_name = os.path.basename(dataset_path)
        solution_filename = f"solution_{base_name}"
        solution_path = os.path.join(SCIPY_OUTPUT_DIR, solution_filename)

        print(f"  Generating solution for {base_name}...")

        # Load points
        try:
            points = np.loadtxt(dataset_path)
            if points.ndim == 1: points = points.reshape(1, -1)
            if points.size == 0: points = points.reshape(0, 2)
        except Exception as e:
            points = np.array([]).reshape(0, 2)

        # --- Run the core analysis logic ---
        solution_points = np.array([]).reshape(0, 2) # Default to empty
        
        try:
            if points.shape[0] < 3:
                # Handle < 3 points (hull is just the points)
                if points.shape[0] == 2:
                    solution_points = points[np.lexsort((points[:, 1], points[:, 0]))]
                elif points.shape[0] == 1:
                    solution_points = points
                # else: 0 points, remains empty
            else:
                # Normal case: compute hull
                hull = ConvexHull(points)
                hull_vertices = points[hull.vertices]
                # Sort for canonical comparison
                solution_points = hull_vertices[np.lexsort((hull_vertices[:, 1], hull_vertices[:, 0]))]

        except Exception as e:
            # Collinear case
            if points.shape[0] > 0:
                min_pt_idx = np.argmin(points[:, 0])
                max_pt_idx = np.argmax(points[:, 0])
                if points[min_pt_idx, 0] == points[max_pt_idx, 0]: # Vertical line
                     min_pt_idx = np.argmin(points[:, 1])
                     max_pt_idx = np.argmax(points[:, 1])
                solution_points = np.array([points[min_pt_idx], points[max_pt_idx]])
                # Sort the two endpoints
                solution_points = solution_points[np.lexsort((solution_points[:, 1], solution_points[:, 0]))]
        
        # --- Save the solution file to the new directory ---
        np.savetxt(solution_path, solution_points, fmt='%.8f', delimiter=' ')
    
    print("--- SciPy Ground Truth Generation Complete ---\n")


# --- 3. C++ Algorithm Batch Runner ---

def run_cpp_algorithms(dataset_paths):
    """
    Runs all compiled C++ executables on all datasets.
    """
    print(f"--- 2. Running All C++ Algorithms ---")
    
    for algo_name, command in ALGO_COMMANDS.items():
        print(f"\n  -- Running: {algo_name} --")
        
        for dataset_path in dataset_paths:
            full_command = command + [dataset_path]
            
            try:
                # Run the C++ program
                # We don't hide output, so we can see its "Processing complete..." messages
                subprocess.run(full_command, check=True, text=True)
            except Exception as e:
                print(f"  [!] FAILED to run '{' '.join(full_command)}'")
                print(f"  [!] Error: {e}")

    print("\n--- C++ Algorithm Runs Complete ---")


# --- 4. Main Execution ---
if __name__ == "__main__":
    
    # Find all datasets
    dataset_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "case_*.txt")))
    
    if not dataset_paths:
        print(f"Error: No datasets found in '{DATASET_DIR}'.")
        print("Run 'python generate_cases.py' first.")
        sys.exit(1)
        
    # Step 1: Generate SciPy solutions
    run_scipy_generation(dataset_paths)
    
    # Step 2: Run all your C++ executables
    run_cpp_algorithms(dataset_paths)
    
    print("\n✅ All processes complete. All output folders should be populated.")