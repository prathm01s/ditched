import subprocess
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import re
import argparse
import pickle
import concurrent.futures
import random
from scipy.spatial import ConvexHull
import dataset_gen as dg

# --- Configuration ---
BIN_DIR = "bin_latest1"
PLOT_BASE_DIR = "plots_latest1"
DATASET_SAVE_DIR = "benchmark_visuals_latest1" 
RESULTS_CACHE_FILE = "benchmark_results_latest1.pkl"
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "."

# OPTIMIZATION: Reduce runs to 5 for speed (statistically sufficient)
NUM_RUNS = 5
# OPTIMIZATION: Strict timeout. If an algo takes >60s, it's failed the "speed" test.
TIMEOUT_SEC = 60
# OPTIMIZATION: Jarvis is O(N^2), strict cutoff.
JARVIS_MAX_N = 25000 

ALGORITHMS_CPP = {
    "Graham Scan": "graham_scan_timed.cpp",
    "Monotone Chain": "monotone_chain_timed.cpp",
    "Optimized Monotone": "optimized_monotone_chain_timed.cpp",
    "QuickHull": "quickhull_timed.cpp",
    "Jarvis March": "jarvis_march_timed.cpp"
}

EXECUTABLES = {name: os.path.join(BIN_DIR, name.replace(" ", "_").lower() + ".out") 
               for name in ALGORITHMS_CPP}

# Range: 1k to 700k
INPUT_SIZES = np.linspace(1000, 700000, 10, dtype=int).tolist()

DIST_DISPLAY_NAMES = {
    "1_uniform_square": "Uniform Square",
    "2_gaussian_blob": "Gaussian Blob",
    "3_multi_cluster_200_instance_1": "Multi-Cluster 200 (1)",
    "3_multi_cluster_200_instance_2": "Multi-Cluster 200 (2)",
    "3_multi_cluster_200_instance_3": "Multi-Cluster 200 (3)",
    "3_multi_cluster_200_instance_4": "Multi-Cluster 200 (4)",
    "3_multi_cluster_200_instance_5": "Multi-Cluster 200 (5)",
    "4_uniform_disk": "Uniform Disk",
    "5_circle_perimeter": "Circle Perimeter",
    "6_rectangle_perimeter": "Rectangle Perimeter",
    "7_log_normal": "Log-Normal",
    "8_laplace": "Laplace",
    "9_pareto": "Pareto",
    "10_parabola": "Parabola",
    "11_fan": "Fan",
    "12_spiral": "Spiral",
    "13_sawtooth": "Sawtooth",
    "14_grid": "Grid",
    "15_needle": "Needle",
    "16_mandelbrot": "Mandelbrot",
    "17_poisson_process": "Poisson Process",
    "18_onion": "Onion"
}

DISTRIBUTIONS = {
    "1_uniform_square": dg.gen_uniform_square,
    "2_gaussian_blob": dg.gen_gaussian_blob,
    "3_multi_cluster_200_instance_1": lambda n: dg.gen_clusters_specific(n, 200),
    "3_multi_cluster_200_instance_2": lambda n: dg.gen_clusters_specific(n, 200),
    "3_multi_cluster_200_instance_3": lambda n: dg.gen_clusters_specific(n, 200),
    "3_multi_cluster_200_instance_4": lambda n: dg.gen_clusters_specific(n, 200),
    "3_multi_cluster_200_instance_5": lambda n: dg.gen_clusters_specific(n, 200),
    "4_uniform_disk": dg.gen_uniform_disk,
    "5_circle_perimeter": dg.gen_circle_perimeter,
    "6_rectangle_perimeter": dg.gen_rectangle_perimeter,
    "7_log_normal": dg.gen_log_normal,
    "8_laplace": dg.gen_laplace,
    "9_pareto": dg.gen_pareto,
    "10_parabola": dg.gen_parabola,
    "11_fan": dg.gen_fan,
    "12_spiral": dg.gen_spiral,
    "13_sawtooth": dg.gen_sawtooth,
    "14_grid": dg.gen_grid,
    "15_needle": dg.gen_needle,
    "16_mandelbrot": dg.gen_mandelbrot,
    "17_poisson_process": dg.gen_poisson_process,
    "18_onion": dg.gen_onion,
}

def ensure_dirs(dist_name=None):
    """
    Create necessary directories for benchmark outputs.
    
    Args:
        dist_name (str, optional): Distribution name for creating subdirectory. Defaults to None.
    
    Returns:
        str: Path to the created plot directory for the distribution, or base plot directory.
    """
    if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)
    if not os.path.exists(PLOT_BASE_DIR): os.makedirs(PLOT_BASE_DIR)
    if not os.path.exists(DATASET_SAVE_DIR): os.makedirs(DATASET_SAVE_DIR)
    if dist_name:
        dist_plot_dir = os.path.join(PLOT_BASE_DIR, dist_name)
        if not os.path.exists(dist_plot_dir): os.makedirs(dist_plot_dir)
        return dist_plot_dir
    return PLOT_BASE_DIR

def save_results(results, filename=RESULTS_CACHE_FILE):
    """
    Save benchmark results to a pickle file for caching and later analysis.
    
    Args:
        results (dict): Nested dictionary containing benchmark results.
        filename (str): Path to output pickle file. Defaults to RESULTS_CACHE_FILE.
    
    Returns:
        None
    """
    print(f"\nSaving benchmark results to {filename}...")
    with open(filename, 'wb') as f: pickle.dump(results, f)

def load_results(filename=RESULTS_CACHE_FILE):
    """
    Load previously saved benchmark results from a pickle file.
    
    Args:
        filename (str): Path to pickle file containing results. Defaults to RESULTS_CACHE_FILE.
    
    Returns:
        dict or None: Loaded benchmark results dictionary, or None if file doesn't exist or loading fails.
    """
    if not os.path.exists(filename): return None
    try:
        with open(filename, 'rb') as f: return pickle.load(f)
    except Exception as e:
        print(f"ERROR loading results: {e}"); return None

def compile_cpp():
    """
    Compile all C++ convex hull algorithm implementations with optimization flags.
    Uses -O3 and -march=native for maximum performance.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        SystemExit: If any compilation fails.
    """
    print("--- Compiling C++ Files ---")
    ensure_dirs()
    for name, src in ALGORITHMS_CPP.items():
        if not os.path.exists(src):
            print(f"ERROR: {src} not found.")
            continue
        exe_path = EXECUTABLES[name]
        cmd = ["g++", "-O3", "-march=native", src, "-o", exe_path]
        try:
            subprocess.run(cmd, check=True)
            print(f"Compiled {name}")
        except subprocess.CalledProcessError:
            print(f"Failed to compile {name}"); exit(1)
    print("Compilation Complete.\n")

def run_cpp_benchmark(exe_path, input_file):
    """
    Execute a C++ convex hull algorithm and measure its performance.
    Uses /usr/bin/time for memory profiling and parses algorithm output for timing.
    
    Args:
        exe_path (str): Path to the compiled executable.
        input_file (str): Path to the input dataset file.
    
    Returns:
        tuple: (time_sec, mem_kb) where time_sec is execution time in seconds (float),
               and mem_kb is peak memory usage in kilobytes (int).
               Returns (None, None) on timeout or execution failure.
    """
    abs_input_path = os.path.abspath(input_file)
    cmd = ["/usr/bin/time", "-v", exe_path, abs_input_path]
    mem_regex = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return None, None
    except Exception: 
        return None, None
    
    if result.returncode != 0: return None, None
    
    time_sec, mem_kb = None, None
    for line in result.stderr.splitlines():
        if "BENCHMARK_TIME_SEC=" in line:
            time_sec = float(line.split("=")[1])
    mem_match = mem_regex.search(result.stderr)
    if mem_match: mem_kb = int(mem_match.group(1))
    return time_sec, mem_kb

def run_scipy_benchmark(points):
    """
    Benchmark SciPy's ConvexHull implementation as a reference.
    
    Args:
        points (numpy.ndarray): Nx2 array of 2D points.
    
    Returns:
        tuple: (time_sec, mem_kb) where time_sec is execution time in seconds (float),
               and mem_kb is peak memory usage in kilobytes (float).
               Returns (0.0, 0.0) if fewer than 3 points.
    """
    if len(points) < 3: return 0.0, 0.0
    tracemalloc.start()
    start_time = time.perf_counter()
    try: hull = ConvexHull(points)
    except Exception: pass 
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end_time - start_time), (peak / 1024)

def benchmark_single_case(task_args):
    """
    Worker function to benchmark a single test case (distribution + input size).
    Generates data, saves files, and runs all algorithms with multiple repetitions.
    Designed to be executed in parallel across multiple processes.
    
    Args:
        task_args (tuple): (dist_name, n) where dist_name is distribution identifier (str)
                          and n is the number of points to generate (int).
    
    Returns:
        tuple: (dist_name, n, case_results, persistent_path) where:
               - dist_name (str): Distribution identifier
               - n (int): Number of points
               - case_results (dict): Nested dict with timing/memory data per algorithm
               - persistent_path (str): Path to saved dataset file
               Returns (dist_name, n, None, None) on failure.
    """
    dist_name, n = task_args
    np.random.seed()
    
    dist_func = DISTRIBUTIONS[dist_name]
    case_results = {name: {'time': [], 'mem': []} for name in ALGORITHMS_CPP}
    case_results['SciPy'] = {'time': [], 'mem': []}
    
    try:
        # 1. Generate
        points = dist_func(n)
        
        # 2. Save Persistent (for visuals)
        persistent_filename = f"{dist_name}_N{n}.txt"
        persistent_path = os.path.join(DATASET_SAVE_DIR, persistent_filename)
        dg.save_points(points, persistent_filename, directory=DATASET_SAVE_DIR)
        
        # 3. Save Temp (for C++)
        input_filename = os.path.join(TEMP_DIR, f"temp_{os.getpid()}_{dist_name}_{n}.txt")
        dg.save_points(points, input_filename, directory=TEMP_DIR)
        
        # 4. Run Algorithms
        for run in range(NUM_RUNS):
            for name, exe_path in EXECUTABLES.items():
                if name == "Jarvis March" and n > JARVIS_MAX_N: continue 
                
                t, m = run_cpp_benchmark(exe_path, input_filename)
                if t is not None:
                    case_results[name]['time'].append(t)
                    case_results[name]['mem'].append(m)
            
            t_scipy, m_scipy = run_scipy_benchmark(points)
            case_results['SciPy']['time'].append(t_scipy)
            case_results['SciPy']['mem'].append(m_scipy)

        if os.path.exists(input_filename): os.remove(input_filename)
        
        # Return path so we can plot it later
        return dist_name, n, case_results, persistent_path

    except Exception as e:
        print(f"ERROR in worker {dist_name} N={n}: {e}")
        return dist_name, n, None, None

def get_means_and_stds(algo_data):
    """
    Calculate mean and standard deviation statistics from algorithm benchmark data.
    
    Args:
        algo_data (dict): Dictionary mapping input sizes (n) to benchmark results.
                         Structure: {n: {'time': [list of times], 'mem': [list of mems]}}
    
    Returns:
        tuple: (ns, times, mems, time_stds, mem_stds) where:
               - ns (numpy.ndarray): Sorted array of input sizes
               - times (numpy.ndarray): Mean execution times for each n
               - mems (numpy.ndarray): Mean memory usage for each n
               - time_stds (numpy.ndarray): Standard deviations of times
               - mem_stds (numpy.ndarray): Standard deviations of memory
    """
    ns = sorted(algo_data.keys())
    times = [np.mean(algo_data[n]['time']) if algo_data[n]['time'] else 0 for n in ns]
    mems = [np.mean(algo_data[n]['mem']) if algo_data[n]['mem'] else 0 for n in ns]
    time_stds = [np.std(algo_data[n]['time']) if algo_data[n]['time'] else 0 for n in ns]
    mem_stds = [np.std(algo_data[n]['mem']) if algo_data[n]['mem'] else 0 for n in ns]
    return np.array(ns), np.array(times), np.array(mems), np.array(time_stds), np.array(mem_stds)

def plot_results(data, dist_name):
    """
    Generate individual and combined performance plots for a distribution.
    Creates time and memory plots for each algorithm, plus combined comparison plots.
    
    Args:
        data (dict): Benchmark results for all algorithms on this distribution.
                    Structure: {algo_name: {n: {'time': [...], 'mem': [...]}}}
        dist_name (str): Distribution identifier for plot titles and file naming.
    
    Returns:
        None. Saves plots to disk in plots_latest1/<dist_name>/ directory.
    """
    save_dir = ensure_dirs(dist_name)
    display_name = DIST_DISPLAY_NAMES.get(dist_name, dist_name)

    # INDIVIDUAL ALGORITHM PLOTS
    for name, algo_data in data.items():
        if name == "SciPy": continue 
        if not algo_data: continue

        ns, means_t, means_m, stds_t, stds_m = get_means_and_stds(algo_data)
        ns_scipy, scipy_t, scipy_m, scipy_stds_t, scipy_stds_m = get_means_and_stds(data['SciPy'])

        # Time
        plt.figure(figsize=(10, 6))
        for n in ns:
            runs = algo_data[n]['time']
            plt.scatter([n]*len(runs), runs, color='blue', alpha=0.15, s=10)
        
        plt.errorbar(ns, means_t, yerr=stds_t, fmt='b-o', label=f'{name} Mean')
        plt.errorbar(ns_scipy, scipy_t, yerr=scipy_stds_t, fmt='g--^', label='SciPy')
        
        plt.xlabel('Input Size N'); plt.ylabel('Time (s)')
        plt.title(f'{name} Time: {display_name}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"{name.replace(' ','_')}_Time.png"), dpi=150)
        plt.close()

        # Memory
        plt.figure(figsize=(10, 6))
        for n in ns:
            runs = algo_data[n]['mem']
            plt.scatter([n]*len(runs), runs, color='orange', alpha=0.15, s=10)
        plt.errorbar(ns, means_m, yerr=stds_m, fmt='o-', color='orange', label=f'{name} Mean')
        plt.errorbar(ns_scipy, scipy_m, yerr=scipy_stds_m, fmt='g--^', label='SciPy')
        
        plt.xlabel('Input Size N'); plt.ylabel('Memory (KB)')
        plt.title(f'{name} Memory: {display_name}')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"{name.replace(' ','_')}_Memory.png"), dpi=150)
        plt.close()

    # COMBINED PLOTS
    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        if not algo_data: continue
        ns, means_t, _, stds_t, _ = get_means_and_stds(algo_data)
        if len(ns) == 0: continue
        plt.errorbar(ns, means_t, yerr=stds_t, label=name, marker='o')
    plt.xlabel('N'); plt.ylabel('Time (s)'); plt.title(f'Time Comparison: {display_name}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "Combined_Time.png"), dpi=150); plt.close()

    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        if not algo_data: continue
        ns, _, means_m, _, stds_m = get_means_and_stds(algo_data)
        if len(ns) == 0: continue
        plt.errorbar(ns, means_m, yerr=stds_m, label=name, marker='o')
    plt.xlabel('N'); plt.ylabel('Memory (KB)'); plt.title(f'Memory Comparison: {display_name}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "Combined_Memory.png"), dpi=150); plt.close()

def plot_grouped_bar_charts(all_results):
    """
    Generate grouped bar chart comparing all algorithms across all distributions.
    Uses the largest input size (N ≈ 700K) for comparison.
    
    Args:
        all_results (dict): Complete benchmark results for all distributions.
                           Structure: {dist_name: {algo_name: {n: {'time': [...], 'mem': [...]}}}}
    
    Returns:
        None. Saves plot to plots_latest1/Grouped_Time_Across_Distributions.png
    """
    print("\n  Generating grouped bar charts...")
    dist_names = sorted(all_results.keys())
    if not dist_names: return
    
    target_n = max(INPUT_SIZES)
    
    time_data = {}; time_err = {}
    valid_dists = []
    
    all_algos = set()
    for d in all_results.values(): all_algos.update(d.keys())
    algo_names = sorted(list(all_algos))
    
    for algo in algo_names:
        time_data[algo] = []
        time_err[algo] = []

    for dist_name in dist_names:
        dist_data = all_results[dist_name]
        
        all_ns = set()
        for ad in dist_data.values(): all_ns.update(ad.keys())
        if not all_ns: continue
        
        closest_n = min(all_ns, key=lambda x: abs(x - target_n))
        if closest_n < target_n * 0.5: continue
            
        valid_dists.append(DIST_DISPLAY_NAMES.get(dist_name, dist_name))
        
        for algo in algo_names:
            if algo in dist_data and closest_n in dist_data[algo]:
                times = dist_data[algo][closest_n]['time']
                time_data[algo].append(np.mean(times) if times else 0)
                time_err[algo].append(np.std(times) if times else 0)
            else:
                time_data[algo].append(0)
                time_err[algo].append(0)
    
    if not valid_dists: return
    
    x = np.arange(len(valid_dists))
    width = 0.8 / len(algo_names)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, algo in enumerate(algo_names):
        if sum(time_data[algo]) == 0: continue
        offset = (i - len(algo_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, time_data[algo], width, yerr=time_err[algo], capsize=3, label=algo)
        ax.bar_label(bars, fmt='%.4f', fontsize=7, rotation=90, padding=3)
    
    ax.set_ylabel('Time (s)'); ax.set_title(f'Time Comparison (N ≈ {target_n:,})')
    ax.set_xticks(x); ax.set_xticklabels(valid_dists, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_BASE_DIR, "Grouped_Time_Across_Distributions.png"), dpi=150)
    plt.close()

def plot_single_algorithm_bars(all_results):
    """
    Generate individual bar charts for each algorithm showing performance across all distributions.
    Uses the largest input size (N ≈ 700K) for comparison.
    
    Args:
        all_results (dict): Complete benchmark results for all distributions.
                           Structure: {dist_name: {algo_name: {n: {'time': [...], 'mem': [...]}}}}
    
    Returns:
        None. Saves plots to plots_latest1/<Algorithm>_Time_All.png for each algorithm.
    """
    print("\n  Generating individual algorithm bar charts...")
    dist_names = sorted(all_results.keys())
    if not dist_names: return
    
    target_n = max(INPUT_SIZES)
    
    all_algos = set()
    for d in all_results.values(): all_algos.update(d.keys())
    
    for algo in all_algos:
        time_vals = []; time_errs = []; valid_dists = []
        
        for dist_name in dist_names:
            dist_data = all_results[dist_name]
            if algo not in dist_data: continue
            algo_data = dist_data[algo]
            if not algo_data: continue
            
            closest_n = min(algo_data.keys(), key=lambda x: abs(x - target_n))
            if closest_n < target_n * 0.5: continue 
            
            times = algo_data[closest_n]['time']
            if times:
                valid_dists.append(DIST_DISPLAY_NAMES.get(dist_name, dist_name))
                time_vals.append(np.mean(times))
                time_errs.append(np.std(times))
        
        if not valid_dists: continue
        
        x = np.arange(len(valid_dists))
        plt.figure(figsize=(14, 6))
        bars = plt.bar(x, time_vals, yerr=time_errs, capsize=4, color='steelblue', alpha=0.8)
        plt.bar_label(bars, fmt='%.5f', fontsize=8, rotation=90, padding=3)
        plt.ylabel('Time (s)'); plt.title(f'{algo}: Time (N ≈ {target_n:,})')
        plt.xticks(x, valid_dists, rotation=45, ha='right'); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_BASE_DIR, f"{algo.replace(' ','_')}_Time_All.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bar-charts-only', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--results-file', default=RESULTS_CACHE_FILE)
    args = parser.parse_args()
    
    if args.bar_charts_only:
        all_res = load_results(args.results_file)
        if all_res:
            plot_grouped_bar_charts(all_res)
            plot_single_algorithm_bars(all_res)
        exit(0)
    
    compile_cpp()
    
    tasks = []
    for dist_name in DISTRIBUTIONS.keys():
        for n in INPUT_SIZES:
            tasks.append((dist_name, n))
            
    random.shuffle(tasks)
            
    # --- SINGLE POOL ARCHITECTURE (NO PLOTTING) ---
    # Pool 1: Benchmarking (Max Cores) - High CPU Priority
    
    # Use all cores for benchmarking
    bench_workers = max(1, os.cpu_count())
    
    print(f"--- Starting Optimized Benchmark ---")
    print(f"    Benchmark Workers: {bench_workers}")
    print(f"    Total Tasks: {len(tasks)}")
    
    start_time = time.time()
    all_distribution_results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=bench_workers) as bench_executor:
        
        # Submit all benchmark tasks
        bench_futures = [bench_executor.submit(benchmark_single_case, task) for task in tasks]
        
        # Track progress
        total = len(bench_futures)
        completed = 0
        
        for future in concurrent.futures.as_completed(bench_futures):
            completed += 1
            if completed % 10 == 0:
                print(f"    Progress: {completed}/{total} benchmarks completed...")
                
            res = future.result()
            if res:
                d_name, n, case_data, _ = res
                
                # 1. Aggregate Data
                if case_data:
                    if d_name not in all_distribution_results: all_distribution_results[d_name] = {}
                    for algo_name, metrics in case_data.items():
                        if metrics['time']:
                            if algo_name not in all_distribution_results[d_name]:
                                all_distribution_results[d_name][algo_name] = {}
                            all_distribution_results[d_name][algo_name][n] = metrics

    print(f"\n--- Benchmark Finished in {time.time() - start_time:.2f} s ---")
    
    print("--- Generating Final Reports ---")
    for dist_name, dist_data in all_distribution_results.items():
        try: plot_results(dist_data, dist_name)
        except Exception as e: print(f"Error plotting {dist_name}: {e}")

    if not args.no_save: save_results(all_distribution_results, args.results_file)
    plot_grouped_bar_charts(all_distribution_results)
    plot_single_algorithm_bars(all_distribution_results)