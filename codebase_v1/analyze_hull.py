import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import sys
import os # Import os to help with creating the new filename

"""
Convex Hull Analysis and Visualization Tool

This script analyzes point datasets by:
- Computing the convex hull using SciPy
- Visualizing all points and hull vertices
- Saving hull vertices to a solution file for verification

Usage:
    python analyze_hull.py <input_file.txt>

Example:
    python analyze_hull.py datasets/case_1.txt
"""

def analyze_and_plot(filename):
    """
    Load points from a file, compute convex hull, and visualize results.
    Saves hull vertices to a solution file for later verification.
    
    Args:
        filename (str): Path to input file containing 2D points (space-separated x y).
    
    Returns:
        None. Displays plot and saves solution file as solution_<input_filename>.
    """
    # Create the output filename
    base_filename = os.path.basename(filename)
    solution_filename = f"solution_{base_filename}"

    # --- 1. Load Data ---
    try:
        points = np.loadtxt(filename)
        if points.ndim == 1: # Handle single point case
            points = points.reshape(1, -1)
            
        if points.shape[0] == 0:
             print(f"Loaded {filename}. File is empty.")
             # Save an empty file
             np.savetxt(solution_filename, points, fmt='%.8f', delimiter=' ')
             print(f"Empty solution file saved to: {solution_filename}")
             plt.title(f"Dataset: {filename} (Empty)")
             plt.show()
             return
             
        if points.shape[0] < 3:
            print(f"Loaded {filename}. Not enough points ({points.shape[0]}) to form a 2D hull.")
            # The "hull" is just the points themselves.
            # Sort them for consistency (lexicographical)
            if points.shape[0] == 2:
                points = points[np.lexsort((points[:, 1], points[:, 0]))]

            np.savetxt(solution_filename, points, fmt='%.8f', delimiter=' ')
            print(f"Solution (the points themselves) saved to: {solution_filename}")
            
            # Still plot the points
            plt.scatter(points[:, 0], points[:, 1], c='blue')
            plt.title(f"Dataset: {filename} (Not enough points for hull)")
            plt.show()
            return

    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return
        
    print(f"Analyzing {filename} ({points.shape[0]} points)...")

    # --- 2. Plot All Points ---
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='All Points')

    # --- 3. Compute, Plot, and SAVE Hull (The "Analysis") ---
    try:
        hull = ConvexHull(points)
        
        # Get the vertices that form the hull
        hull_vertices = points[hull.vertices]
        
        # Sort vertices lexicographically for consistent output
        hull_vertices = hull_vertices[np.lexsort((hull_vertices[:, 1], hull_vertices[:, 0]))]
        
        print(f"Analysis complete. Hull has {len(hull_vertices)} vertices.")
        
        # Save the hull vertices to a file
        np.savetxt(solution_filename, hull_vertices, fmt='%.8f', delimiter=' ')
        print(f"Hull vertices saved to: {solution_filename}")
        
        # Plot hull vertices as large red circles
        plt.scatter(hull_vertices[:, 0], hull_vertices[:, 1], c='red', s=40, label='Hull Vertices')
        
        # Draw the hull boundary (using the unsorted points from hull.vertices for correct plotting order)
        plot_hull_points = points[hull.vertices]
        plot_hull = np.append(plot_hull_points, [plot_hull_points[0]], axis=0)
        
        plt.plot(plot_hull[:, 0], plot_hull[:, 1], 'r--', lw=2, label='Correct Hull Boundary')

    except Exception as e:
        # This typically happens if all points are collinear
        print(f"Could not compute Convex Hull (Scipy error): {e}")
        print("This is a degenerate case (e.g., all points are collinear).")
        
        if points.shape[0] > 0:
            # Find the two extreme points of the line
            min_pt_idx = np.argmin(points[:, 0])
            max_pt_idx = np.argmax(points[:, 0])
            
            # Handle vertical line case
            if points[min_pt_idx, 0] == points[max_pt_idx, 0]: 
                 min_pt_idx = np.argmin(points[:, 1])
                 max_pt_idx = np.argmax(points[:, 1])
            
            hull_line = np.array([points[min_pt_idx], points[max_pt_idx]])
            
            # Save the two endpoints to a file
            np.savetxt(solution_filename, hull_line, fmt='%.8f', delimiter=' ')
            print(f"Degenerate hull (line segment) saved to: {solution_filename}")
            np.savetxt(solution_filename, hull_line, fmt='%.8f', delimiter=' ')
            print(f"Hull endpoints saved to: {solution_filename}")
            
            plt.plot(hull_line[:, 0], hull_line[:, 1], 'r--', lw=2, label='Correct Hull (Line)')


    # --- 4. Show Plot ---
    plt.title(f"Analysis of: {filename}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Important for geometry
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_hull.py <filename.txt>")
        sys.exit(1)
        
    filename = sys.argv[1]
    analyze_and_plot(filename)