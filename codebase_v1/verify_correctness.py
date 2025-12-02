import numpy as np
import os
import glob
import sys
from scipy.spatial import ConvexHull

"""
Convex Hull Correctness Verification Tool

This script validates C++ algorithm implementations by:
1. Generating ground truth solutions using SciPy's ConvexHull
2. Comparing C++ output files against ground truth
3. Reporting mismatches and verification results

Usage:
    python verify_correctness.py

The script automatically processes all datasets in datasets/ directory
and compares outputs in configured algorithm output folders against SciPy ground truth.

Configuration:
    - Edit ALGO_FOLDERS to match your C++ output directory names
    - Datasets expected in DATASET_DIR (default: "datasets")
    - Ground truth saved to SOLUTION_DIR (default: "SCIPY_OUTPUT")
"""

# --- 1. Configuration ---

# Folders where your C++ programs save their results
ALGO_FOLDERS = {
    "Monotone (Base)": "MONOTONE_OUTPUT",
    "Monotone (Optimized)": "OPTIMIZED_MONOTONE_OUTPUT",
    "QuickHull (C++)": "QUICKHULL_OUTPUT"
}

# Folder where your datasets are
DATASET_DIR = "datasets"

# Folder where this script will save the SciPy ground truth
SOLUTION_DIR = "SCIPY_OUTPUT"


# --- 2. Helper Function (Same as before) ---

def load_and_sort_points(filepath):
    """
    Load points from a .txt file and return them lexicographically sorted.
    
    Args:
        filepath (str): Path to the points file (space-separated x y coordinates).
    
    Returns:
        numpy.ndarray: Sorted 2D array of points (shape: [n, 2]), or None if file not found.
        Returns empty array (shape: [0, 2]) if file is empty or malformed.
    
    Note:
        Points are sorted first by x-coordinate, then by y-coordinate.
    """
    if not os.path.exists(filepath):
        return None
        
    try:
        points = np.loadtxt(filepath)
    except Exception as e:
        return np.array([]).reshape(0, 2)

    if points.size == 0:
        return np.array([]).reshape(0, 2)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.size == 0:
        return np.array([]).reshape(0, 2)

    # Sort the points: first by x-coordinate, then by y-coordinate
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    return points[sorted_indices]


# --- 3. Ground Truth Generation Function ---

def generate_scipy_solutions():
    """
    Generate SciPy-based ground truth convex hull solutions for all datasets.
    
    This function:
    - Creates the SOLUTION_DIR if it doesn't exist
    - Processes all case_*.txt files in DATASET_DIR
    - Computes convex hull using SciPy's ConvexHull
    - Saves sorted hull vertices as solution_<dataset_name>.txt
    
    Returns:
        bool: False if no datasets found, True otherwise.
    
    Note:
        - Skips datasets that already have solutions (won't overwrite)
        - Handles edge cases: < 3 points, collinear points, empty datasets
        - Hull vertices are sorted lexicographically for consistent comparison
    """
    print(f"--- 1. Generating SciPy Ground Truth in '{SOLUTION_DIR}' ---")
    os.makedirs(SOLUTION_DIR, exist_ok=True)
    dataset_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "case_*.txt")))

    if not dataset_paths:
        print(f"Error: No datasets found in '{DATASET_DIR}'.")
        print("Run 'python generate_cases.py' first.")
        return False

    for dataset_path in dataset_paths:
        base_name = os.path.basename(dataset_path)
        solution_filename = f"solution_{base_name}"
        solution_path = os.path.join(SOLUTION_DIR, solution_filename)

        # Don't re-generate if it already exists
        if os.path.exists(solution_path):
            continue

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
            print(f"    - Note: SciPy error (e.g., collinear), finding extremes.")
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
    return True


# --- 4. Main Verification Logic (Now points to SOLUTION_DIR) ---

def run_verification():
    """
    Compares all algorithm outputs against the ground truth
    files in SOLUTION_DIR.
    """
    print(f"--- 2. Starting Output Verification (Comparing against '{SOLUTION_DIR}') ---")
    
    dataset_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "case_*.txt")))
    
    if not dataset_paths:
        print(f"Error: No datasets found in '{DATASET_DIR}'.")
        return

    summary = {name: {"correct": 0, "mismatch": 0, "missing": 0} 
               for name in ALGO_FOLDERS.keys()}

    # --- Loop 1: Iterate over each test case ---
    for dataset_path in dataset_paths:
        base_name = os.path.basename(dataset_path)
        print(f"\n--- Verifying Case: {base_name} ---")
        
        # 1. Find and load the Ground Truth solution
        solution_filename = f"solution_{base_name}"
        solution_path = os.path.join(SOLUTION_DIR, solution_filename) # <-- Uses new path
        
        truth_points = load_and_sort_points(solution_path)
        
        if truth_points is None:
            print(f"  [!] CRITICAL ERROR: Ground truth file missing at '{solution_path}'")
            print("      This should not happen. Check permissions.")
            continue
            
        # --- Loop 2: Iterate over each of your algorithms ---
        for algo_name, output_folder in ALGO_FOLDERS.items():
            
            # 2. Find the algorithm's corresponding output file
            algo_output_base = base_name.replace(".txt", "_output.txt")
            algo_output_path = os.path.join(output_folder, algo_output_base)
            
            algo_points = load_and_sort_points(algo_output_path)
            
            # 3. Compare Truth vs. Algorithm Output
            if algo_points is None:
                print(f"  ❌ {algo_name}: FAILED (Missing File)")
                print(f"     - Expected file at: {algo_output_path}")
                summary[algo_name]["missing"] += 1
                continue
            
            try:
                # 4. The actual comparison!
                if algo_points.shape == truth_points.shape and \
                   np.allclose(algo_points, truth_points, atol=1e-6):
                    
                    print(f"  ✅ {algo_name}: CORRECT")
                    summary[algo_name]["correct"] += 1
                
                else:
                    print(f"  ❌ {algo_name}: FAILED (Mismatch)")
                    summary[algo_name]["mismatch"] += 1
                    
                    if algo_points.shape != truth_points.shape:
                        print(f"     - Vertex Count Mismatch:")
                        print(f"       - Expected Shape: {truth_points.shape}")
                        print(f"       - Got Shape:      {algo_points.shape}")
                    else:
                        print(f"     - Coordinate Mismatch (after sorting):")
                        print(f"       - Expected Points:\n{truth_points}")
                        print(f"       - Got Points:\n{algo_points}")
                        
            except Exception as e:
                print(f"  ❌ {algo_name}: FAILED (Error during comparison: {e})")
                summary[algo_name]["mismatch"] += 1


    # --- 5. Final Summary Report ---
    print("\n\n" + "="*30)
    print("  ✅ FINAL VERIFICATION SUMMARY ✅")
    print("="*30 + "\n")
    
    for algo_name, stats in summary.items():
        print(f"--- Algorithm: {algo_name} ---")
        print(f"  Correct:  {stats['correct']}")
        print(f"  Mismatch: {stats['mismatch']}")
        print(f"  Missing:  {stats['missing']}")
        total = stats['correct'] + stats['mismatch'] + stats['missing']
        if total > 0:
            pass_rate = (stats['correct'] / total) * 100
            print(f"  Pass Rate: {pass_rate:.1f}%")
        print("")

# --- 6. Run Both Steps ---
if __name__ == "__main__":
    if generate_scipy_solutions():
        run_verification()