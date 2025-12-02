import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import glob
import os
import re  
import tracemalloc  
from scipy.spatial import ConvexHull

# --- 1. CONFIGURATION --- (Unchanged)
MY_ALGO_COMMAND = ["./monotone_chain"]
MY_OPTIMIZED_ALGO_COMMAND = ["./optimized_monotone_chain"]
DATASET_DIR = "datasets"

# --- 2. run_benchmarks() FUNCTION --- (Unchanged)
# This function is already correct. It records 'None' on failures,
# which is what we want.
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
            # This 'continue' is new, to prevent errors on empty files
            # But your C++ error handling already catches it, so this is just safer.
            if not np.any(points):
                is_runnable = False


        # --- A: Time/Memory for (Base) Monotone Chain ---
        command_base = ["/usr/bin/time", "-v"] + MY_ALGO_COMMAND + [filepath]
        time_c = None
        mem_c = None
        try:
            start_time_c = time.perf_counter()
            result = subprocess.run(command_base, check=True, capture_output=True, text=True)
            end_time_c = time.perf_counter()
            time_c = end_time_c - start_time_c
            mem_match = mem_regex.search(result.stderr)
            if mem_match:
                mem_c = int(mem_match.group(1)) 

        except Exception as e:
            print(f"  ERROR running your BASE algorithm: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)

        # --- B: Time/Memory for (Optimized) Monotone Chain ---
        command_optimized = ["/usr/bin/time", "-v"] + MY_OPTIMIZED_ALGO_COMMAND + [filepath]
        time_c_optimized = None
        mem_c_optimized = None
        try:
            start_time_c_opt = time.perf_counter()
            result_opt = subprocess.run(command_optimized, check=True, capture_output=True, text=True)
            end_time_c_opt = time.perf_counter()
            time_c_optimized = end_time_c_opt - start_time_c_opt
            
            mem_match_opt = mem_regex.search(result_opt.stderr)
            if mem_match_opt:
                mem_c_optimized = int(mem_match_opt.group(1))

        except Exception as e:
            print(f"  ERROR running your OPTIMIZED algorithm: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)

        # --- C: Time/Memory for SciPy (Quickhull) ---
        time_scipy = None
        mem_scipy = None
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
                # Still record the time it took to fail
                time_scipy = time.perf_counter() - start_time_scipy 
                # But memory is None (it failed)
                mem_scipy = None 
                print(f"  Note: SciPy raised an error (e.g., collinear): {e}")
        
        results.append({
            "name": filename.replace("case_", "").replace(".txt", ""),
            "base_time": time_c,
            "opt_time": time_c_optimized,
            "scipy_time": time_scipy,
            "base_mem": mem_c,
            "opt_mem": mem_c_optimized,
            "scipy_mem": mem_scipy
        })

    print("--- Benchmark Complete ---")
    return results

# --- 3. THE PLOTTING FUNCTIONS ---

# --- !!! MODIFIED plot_time_results !!! ---
def plot_time_results(results):
    # Filter out only cases where YOUR code failed
    valid_results = [r for r in results if r["base_time"] is not None and 
                     r["opt_time"] is not None]
    
    if not valid_results:
        print("No valid TIME results to plot. Did all your C++ runs fail?")
        return

    labels = [r["name"] for r in valid_results]
    base_times = [r["base_time"] for r in valid_results]
    opt_times = [r["opt_time"] for r in valid_results]
    # Replace None with np.nan so Matplotlib can plot a gap
    scipy_times = [r["scipy_time"] if r["scipy_time"] is not None else np.nan for r in valid_results]


    x = np.arange(len(labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(17, 10)) 
    rects1 = ax.bar(x - width, base_times, width, label='My Algo (Base)')
    rects2 = ax.bar(x, opt_times, width, label='My Algo (Optimized)')
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

# --- !!! MODIFIED plot_memory_results !!! ---
def plot_memory_results(results):
    # Filter out only cases where YOUR code failed
    valid_results = [r for r in results if r["base_mem"] is not None and 
                     r["opt_mem"] is not None]
    
    if not valid_results:
        print("No valid MEMORY results to plot. Did all your C++ runs fail?")
        return

    labels = [r["name"] for r in valid_results]
    base_mems = [r["base_mem"] for r in valid_results]
    opt_mems = [r["opt_mem"] for r in valid_results]
    # Replace None with np.nan so Matplotlib can plot a gap
    scipy_mems = [r["scipy_mem"] if r["scipy_mem"] is not None else np.nan for r in valid_results]

    x = np.arange(len(labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(17, 10)) 
    rects1 = ax.bar(x - width, base_mems, width, label='My Algo (Base)')
    rects2 = ax.bar(x, opt_mems, width, label='My Algo (Optimized)')
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


# --- 4. MAIN EXECUTION --- (Unchanged)
if __name__ == "__main__":
    benchmark_results = run_benchmarks()
    if benchmark_results:
        plot_time_results(benchmark_results)    
        plot_memory_results(benchmark_results)