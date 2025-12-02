import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import glob
import os
import re
import sys  # <--- NEW
import tracemalloc
from scipy.spatial import ConvexHull

"""
Benchmarking Script for Convex Hull Algorithms

This script benchmarks multiple convex hull algorithm implementations by:
- Running C++ executables on test datasets
- Measuring execution time and memory usage
- Comparing against SciPy's ConvexHull implementation
- Generating comparative plots

Usage:
    python benchmark.py ./executable1 ./executable2 ...

Example:
    python benchmark.py ./monotone_chain ./quickhull ./graham_scan
"""

# --- 1. Configuration ---
DATASET_DIR = "datasets"  # Directory containing test datasets

# --- 2. THE BENCHMARKING FUNCTION ---

def run_benchmarks(algo_map):
    """
    Run benchmarks on all datasets for specified C++ algorithms plus SciPy.
    
    Args:
        algo_map (dict): Dictionary mapping algorithm names to command arrays.
                        Format: {"algorithm_name": ["./executable_path"]}
    
    Returns:
        list: List of dictionaries containing benchmark results for each dataset.
              Each dict has keys: "name", "{algo}_time", "{algo}_mem", "scipy_time", "scipy_mem"
    """
    dataset_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
    if not dataset_files:
        print(f"Error: No datasets found in '{DATASET_DIR}'.")
        return []

    results = []
    print("--- Starting Benchmark (Time and Memory) ---")
    mem_regex = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

    for filepath in dataset_files:
        filename = os.path.basename(filepath)
        print(f"Benchmarking: {filename}...")
        
        result_row = {"name": filename.replace("case_", "").replace(".txt", "")}
        
        try:
            points = np.loadtxt(filepath)
            if points.ndim == 1: points = points.reshape(1, -1)
            is_runnable_scipy = points.shape[0] >= 3
        except Exception:
            is_runnable_scipy = False

        # --- A: Run all C++ Algorithms passed from Makefile ---
        for name, command in algo_map.items():
            full_command = ["/usr/bin/time", "-v"] + command + [filepath]
            time_val, mem_val = None, None
            try:
                start_time = time.perf_counter()
                result = subprocess.run(full_command, check=True, capture_output=True, text=True)
                time_val = time.perf_counter() - start_time
                mem_match = mem_regex.search(result.stderr)
                if mem_match: 
                    mem_val = int(mem_match.group(1))
            except Exception as e:
                print(f"  ERROR running {name}: {e}")
                if hasattr(e, 'stderr'): print(e.stderr)
            
            result_row[f"{name}_time"] = time_val
            result_row[f"{name}_mem"] = mem_val

        # --- B: Run SciPy (Quickhull) ---
        time_scipy, mem_scipy = None, None
        if is_runnable_scipy:
            try:
                tracemalloc.start()
                start_time_scipy = time.perf_counter()
                hull = ConvexHull(points)
                end_time_scipy = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                time_scipy = end_time_scipy - start_time_scipy
                mem_scipy = peak / 1024  # Convert bytes to Kilobytes (KiB)
            except Exception as e:
                tracemalloc.stop()
                time_scipy = time.perf_counter() - start_time_scipy
                print(f"  Note: SciPy raised an error (e.g., collinear): {e}")
        
        result_row["scipy_time"] = time_scipy
        result_row["scipy_mem"] = mem_scipy
        
        results.append(result_row)

    print("--- Benchmark Complete ---")
    return results

# --- 3. DYNAMIC PLOTTING FUNCTIONS ---

def plot_results(results, algo_names, metric_key, y_label, title, filename, fmt):
    """
    Generate bar chart comparing algorithm performance for either time or memory.
    
    Args:
        results (list): Benchmark results from run_benchmarks().
        algo_names (list): List of algorithm names to plot.
        metric_key (str): Metric to plot - either 'time' or 'mem'.
        y_label (str): Y-axis label for the plot.
        title (str): Plot title.
        filename (str): Output filename for the saved plot.
        fmt (str): Format string for bar labels (e.g., '%.5f' for time, '%.0f' for memory).
    
    Returns:
        None. Saves plot to filename and displays it.
    """
    
    # Filter out results where any of the C++ algos failed
    valid_results = [r for r in results if all(r.get(f"{name}_{metric_key}") is not None for name in algo_names)]
    
    if not valid_results:
        print(f"No valid {metric_key.upper()} results to plot. Did all C++ runs fail?")
        return

    labels = [r["name"] for r in valid_results]
    x = np.arange(len(labels))
    
    # Total number of bars
    num_bars = len(algo_names) + 1 
    width = 0.8 / num_bars  # Dynamic width
    
    fig, ax = plt.subplots(figsize=(17, 10))

    # Function to calculate the position for the i-th bar
    def get_bar_position(i):
        # This centers the group of bars over the x-tick
        return x - (num_bars - 1) * 0.5 * width + i * width

    # Plot each C++ algorithm dynamically
    for i, name in enumerate(algo_names):
        data = [r[f"{name}_{metric_key}"] for r in valid_results]
        pos = get_bar_position(i)
        rects = ax.bar(pos, data, width, label=name)
        ax.bar_label(rects, padding=3, fmt=fmt, rotation=90, fontsize=8)

    # Plot SciPy (always the last bar)
    scipy_data = [r[f"scipy_{metric_key}"] if r.get(f"scipy_{metric_key}") is not None else np.nan for r in valid_results]
    pos = get_bar_position(num_bars - 1)
    rects_scipy = ax.bar(pos, scipy_data, width, label='SciPy (Quickhull)')
    ax.bar_label(rects_scipy, padding=3, fmt=fmt, rotation=90, fontsize=8)

    # --- Formatting ---
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Benchmark plot saved to '{filename}'")
    plt.show()

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    """
    Main execution entry point.
    Parses command-line arguments for algorithm executables,
    runs benchmarks, and generates comparative plots.
    """
    
    # Read executable paths from command line (e.g., ["./monotone_chain", "./quickhull"])
    executable_paths = sys.argv[1:]
    
    if not executable_paths:
        print("Error: No algorithms specified.")
        print("Usage: python benchmark.py ./path/to/algo1 ./path/to/algo2")
        sys.exit(1)

    # Create the map of {"clean_name": ["/path/to/executable"]}
    algo_map = {os.path.basename(p): [p] for p in executable_paths}
    algo_names = list(algo_map.keys())

    # Run the benchmarks
    benchmark_results = run_benchmarks(algo_map)
    
    if benchmark_results:
        # Plot Time
        plot_results(benchmark_results, algo_names, 
                     metric_key='time', 
                     y_label='Time (seconds, log scale)', 
                     title='Algorithm Performance: TIME', 
                     filename='benchmark_plot_TIME.png',
                     fmt='%.5f')
                     
        # Plot Memory
        plot_results(benchmark_results, algo_names, 
                     metric_key='mem', 
                     y_label='Memory (Kilobytes, log scale)', 
                     title='Algorithm Performance: MEMORY', 
                     filename='benchmark_plot_MEMORY.png',
                     fmt='%.0f')