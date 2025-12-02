import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import glob
import os
import re  
import tracemalloc  
from scipy.spatial import ConvexHull

# --- 1. CONFIGURATION ---
# We now have THREE algorithms to test
MY_ALGO_COMMAND = ["./monotone_chain"]
MY_QUICKHULL_COMMAND = ["./quickhull"] # <--- Renamed as requested
DATASET_DIR = "datasets"

# --- 2. THE BENCHMARKING FUNCTION ---

def run_benchmarks():
    dataset_files = glob.glob(os.path.join(DATASET_DIR, "*.txt"))
    dataset_files.sort()
    
    if not dataset_files:
        print(f"Error: No datasets found in '{DATASET_DIR}'.")
        return

    results = []
    print("--- Starting Benchmark (Time and Memory) ---")
    mem_regex = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

    for filepath in dataset_files:
        filename = os.path.basename(filepath)
        print(f"Benchmarking: {filename}...")
        
        try:
            points = np.loadtxt(filepath)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            is_runnable = points.shape[0] >= 3
        except Exception as e:
            print(f"  Skipping (empty or corrupt file): {e}")
            is_runnable = False
            if not np.any(points):
                is_runnable = False

        # --- A: Time/Memory for (Base) Monotone Chain ---
        command_base = ["/usr/bin/time", "-v"] + MY_ALGO_COMMAND + [filepath]
        time_c, mem_c = None, None
        try:
            start_time = time.perf_counter()
            result = subprocess.run(command_base, check=True, capture_output=True, text=True)
            time_c = time.perf_counter() - start_time
            mem_match = mem_regex.search(result.stderr)
            if mem_match: mem_c = int(mem_match.group(1)) 
        except Exception as e:
            print(f"  ERROR running BASE Monotone: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)

        # --- B: Time/Memory for (Your) C++ QuickHull ---
        command_quickhull = ["/usr/bin/time", "-v"] + MY_QUICKHULL_COMMAND + [filepath]
        time_c_quickhull, mem_c_quickhull = None, None
        try:
            start_time = time.perf_counter()
            result_qh = subprocess.run(command_quickhull, check=True, capture_output=True, text=True)
            time_c_quickhull = time.perf_counter() - start_time
            mem_match_qh = mem_regex.search(result_qh.stderr)
            if mem_match_qh: mem_c_quickhull = int(mem_match_qh.group(1))
        except Exception as e:
            print(f"  ERROR running C++ QuickHull: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)

        # --- C: Time/Memory for SciPy (Quickhull) ---
        time_scipy, mem_scipy = None, None
        if is_runnable:
            try:
                tracemalloc.start()  
                start_time_scipy = time.perf_counter()
                hull = ConvexHull(points) 
                end_time_scipy = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory() 
                tracemalloc.stop() 
                time_scipy = end_time_scipy - start_time_scipy
                mem_scipy = peak / 1024  
            except Exception as e:
                tracemalloc.stop() 
                time_scipy = time.perf_counter() - start_time_scipy 
                mem_scipy = None 
                print(f"  Note: SciPy raised an error (e.g., collinear): {e}")
        
        results.append({
            "name": filename.replace("case_", "").replace(".txt", ""),
            "base_time": time_c,
            "qh_time": time_c_quickhull,
            "scipy_time": time_scipy,
            "base_mem": mem_c,
            "qh_mem": mem_c_quickhull,
            "scipy_mem": mem_scipy
        })

    print("--- Benchmark Complete ---")
    return results

# --- 3. THE PLOTTING FUNCTIONS ---

def plot_time_results(results):
    # Filter out cases where ANY of your C++ code failed
    valid_results = [r for r in results if r["base_time"] is not None and 
                     r["qh_time"] is not None]
    
    if not valid_results:
        print("No valid TIME results to plot. Did all your C++ runs fail?")
        return

    labels = [r["name"] for r in valid_results]
    base_times = [r["base_time"] for r in valid_results]
    qh_times = [r["qh_time"] for r in valid_results]
    scipy_times = [r["scipy_time"] if r["scipy_time"] is not None else np.nan for r in valid_results]

    x = np.arange(len(labels))
    width = 0.25  # <--- Adjusted width for 3 bars
    
    fig, ax = plt.subplots(figsize=(17, 10))
    rects1 = ax.bar(x - width, base_times, width, label='Monotone (Base)')
    rects2 = ax.bar(x, qh_times, width, label='My QuickHull (C++)')
    rects3 = ax.bar(x + width, scipy_times, width, label='SciPy (Quickhull)')

    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('Algorithm Performance: TIME', fontsize=16) 
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10) 
    ax.legend(fontsize=12) 

    ax.bar_label(rects1, padding=3, fmt='%.5f', rotation=90, fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.5f', rotation=90, fontsize=8)
    ax.bar_label(rects3, padding=3, fmt='%.5f', rotation=90, fontsize=8)
    
    ax.set_yscale('log')
    fig.tight_layout()
    plt.savefig("benchmark_plot_TIME.png") 
    print("\nBenchmark TIME plot saved to 'benchmark_plot_TIME.png'")
    plt.show()

def plot_memory_results(results):
    # Filter out cases where ANY of your C++ code failed
    valid_results = [r for r in results if r["base_mem"] is not None and 
                     r["qh_mem"] is not None]
    
    if not valid_results:
        print("No valid MEMORY results to plot. Did all your C++ runs fail?")
        return

    labels = [r["name"] for r in valid_results]
    base_mems = [r["base_mem"] for r in valid_results]
    qh_mems = [r["qh_mem"] for r in valid_results]
    scipy_mems = [r["scipy_mem"] if r["scipy_mem"] is not None else np.nan for r in valid_results]

    x = np.arange(len(labels))
    width = 0.25  # <--- Adjusted width for 3 bars

    fig, ax = plt.subplots(figsize=(17, 10))
    rects1 = ax.bar(x - width, base_mems, width, label='Monotone (Base)')
    rects2 = ax.bar(x, qh_mems, width, label='My QuickHull (C++)')
    rects3 = ax.bar(x + width, scipy_mems, width, label='SciPy (Quickhull)')

    ax.set_ylabel('Memory (Kilobytes, log scale)')
    ax.set_title('Algorithm Performance: MEMORY', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)

    ax.bar_label(rects1, padding=3, fmt='%.0f', rotation=90, fontsize=8)
    ax.bar_label(rects2, padding=3, fmt='%.0f', rotation=90, fontsize=8)
    ax.bar_label(rects3, padding=3, fmt='%.0f', rotation=90, fontsize=8)
    
    ax.set_yscale('log') 
    fig.tight_layout()
    plt.savefig("benchmark_plot_MEMORY.png") 
    print("Benchmark MEMORY plot saved to 'benchmark_plot_MEMORY.png'")
    plt.show()

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    if benchmark_results:
        plot_time_results(benchmark_results)    
        plot_memory_results(benchmark_results)