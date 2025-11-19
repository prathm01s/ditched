import subprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import re
import argparse
import pickle
from scipy.spatial import ConvexHull
import dataset_gen as dg

# --- Configuration ---
BIN_DIR = "bin"
PLOT_BASE_DIR = "plots"
RESULTS_CACHE_FILE = "benchmark_results.pkl"  # Cache file for storing results

ALGORITHMS_CPP = {
    "Graham Scan": "graham_scan_timed.cpp",
    "Monotone Chain": "monotone_chain_timed.cpp",
    "Optimized Monotone": "optimized_monotone_chain_timed.cpp",
    "QuickHull": "quickhull_timed.cpp"
}
EXECUTABLES = {name: os.path.join(BIN_DIR, name.replace(" ", "_").lower() + ".out") 
               for name in ALGORITHMS_CPP}

# Range: 10 steps from 1k to 1M
INPUT_SIZES = np.linspace(1000, 1000000, 10, dtype=int).tolist()
NUM_RUNS = 10  # Runs per N

# Friendly distribution names for plots
DIST_DISPLAY_NAMES = {
    "1_uniform_square": "Uniform Square",
    "2_gaussian_blob": "Gaussian Blob",
    "3_multi_cluster_200_instance_1": "Multi-Cluster 200 (Instance 1)",
    "3_multi_cluster_200_instance_2": "Multi-Cluster 200 (Instance 2)",
    "3_multi_cluster_200_instance_3": "Multi-Cluster 200 (Instance 3)",
    "3_multi_cluster_200_instance_4": "Multi-Cluster 200 (Instance 4)",
    "3_multi_cluster_200_instance_5": "Multi-Cluster 200 (Instance 5)",
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

# Map of all distributions in dataset_gen.py
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
    if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)
    if not os.path.exists(PLOT_BASE_DIR): os.makedirs(PLOT_BASE_DIR)
    
    if dist_name:
        dist_plot_dir = os.path.join(PLOT_BASE_DIR, dist_name)
        if not os.path.exists(dist_plot_dir):
            os.makedirs(dist_plot_dir)
        return dist_plot_dir
    return PLOT_BASE_DIR

def save_results(results, filename=RESULTS_CACHE_FILE):
    """Save benchmark results to a pickle file for later use"""
    print(f"\nSaving benchmark results to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved successfully.")

def load_results(filename=RESULTS_CACHE_FILE):
    """Load benchmark results from a pickle file"""
    if not os.path.exists(filename):
        return None
    
    print(f"Loading benchmark results from {filename}...")
    try:
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"Results loaded successfully ({len(results)} distributions).")
        return results
    except Exception as e:
        print(f"ERROR loading results: {e}")
        return None

def compile_cpp():
    print("--- Compiling C++ Files ---")
    ensure_dirs()
    for name, src in ALGORITHMS_CPP.items():
        if not os.path.exists(src):
            print(f"ERROR: {src} not found.")
            continue
        exe_path = EXECUTABLES[name]
        cmd = ["g++", "-O3", src, "-o", exe_path]
        try:
            subprocess.run(cmd, check=True)
            print(f"Compiled {name}")
        except subprocess.CalledProcessError:
            print(f"Failed to compile {name}")
            exit(1)
    print("Compilation Complete.\n")

def run_cpp_benchmark(exe_path, input_file):
    """Run C++ benchmark with accurate memory measurement using /usr/bin/time -v"""
    abs_input_path = os.path.abspath(input_file)
    cmd = ["/usr/bin/time", "-v", exe_path, abs_input_path]
    
    mem_regex = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        return None, None

    if result.returncode != 0:
        return None, None
    
    time_sec, mem_kb = None, None
    
    # Parse time from stderr (BENCHMARK_TIME_SEC= is printed to stderr by C++ programs)
    # Parse memory from stderr (/usr/bin/time -v output also goes to stderr)
    stderr_lines = result.stderr.splitlines()
    
    for line in stderr_lines:
        if "BENCHMARK_TIME_SEC=" in line:
            time_sec = float(line.split("=")[1])
    
    # Parse memory from /usr/bin/time output
    mem_match = mem_regex.search(result.stderr)
    if mem_match:
        mem_kb = int(mem_match.group(1))
    
    return time_sec, mem_kb

def run_scipy_benchmark(points):
    if len(points) < 3: return 0.0, 0.0
    tracemalloc.start()
    start_time = time.perf_counter()
    try:
        hull = ConvexHull(points)
    except Exception:
        pass # Handle degenerate cases gracefully
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end_time - start_time), (peak / 1024)

def benchmark_distribution(dist_name, dist_func):
    """Runs benchmarks for a specific distribution."""
    print(f"\n=== Benchmarking Distribution: {dist_name} ===")
    
    data = {name: {} for name in ALGORITHMS_CPP}
    data['SciPy'] = {}

    total_steps = len(INPUT_SIZES)
    for idx, n in enumerate(INPUT_SIZES):
        print(f"  [N={n:<8}] ({idx+1}/{total_steps})")
        
        # Generate data
        points = dist_func(n)
        # Handle cases where generator produces fewer points (e.g. Mandelbrot)
        actual_n = len(points)
        
        input_filename = f"temp_input_{dist_name}_{n}.txt"
        dg.save_points(points, input_filename, directory=".")
        
        for name in data:
            data[name][actual_n] = {'time': [], 'mem': []}

        for run in range(NUM_RUNS):
            # Run C++ Algos
            for name, exe_path in EXECUTABLES.items():
                t, m = run_cpp_benchmark(exe_path, input_filename)
                if t is not None:
                    data[name][actual_n]['time'].append(t)
                    data[name][actual_n]['mem'].append(m)
            
            # Run SciPy
            t_scipy, m_scipy = run_scipy_benchmark(points)
            data['SciPy'][actual_n]['time'].append(t_scipy)
            data['SciPy'][actual_n]['mem'].append(m_scipy)

        # --- Debug: print counts / mean / stddev for each algorithm at this N ---
        try:
            print(f"  Debug stats for N={actual_n}:")
            for name in data:
                times = data[name][actual_n]['time']
                mems = data[name][actual_n]['mem']
                count = len(times)
                mean_t = np.mean(times) if count else 0.0
                std_t = np.std(times) if count else 0.0
                mean_m = np.mean(mems) if len(mems) else 0.0
                std_m = np.std(mems) if len(mems) else 0.0
                print(f"    {name:15} count={count:<2} mean_time={mean_t:.6f}s std_time={std_t:.6f}s mean_mem={mean_m:.2f}KB std_mem={std_m:.2f}KB")
        except Exception as _e:
            # Non-fatal: debugging should not stop benchmarking
            print(f"  Debug printing failed: {_e}")

        if os.path.exists(input_filename):
            os.remove(input_filename)
            
    return data

def get_means_and_stds(algo_data):
    """Get means and standard deviations for plotting with error bars"""
    ns = sorted(algo_data.keys())
    times = [np.mean(algo_data[n]['time']) if algo_data[n]['time'] else 0 for n in ns]
    mems = [np.mean(algo_data[n]['mem']) if algo_data[n]['mem'] else 0 for n in ns]
    time_stds = [np.std(algo_data[n]['time']) if algo_data[n]['time'] else 0 for n in ns]
    mem_stds = [np.std(algo_data[n]['mem']) if algo_data[n]['mem'] else 0 for n in ns]
    return np.array(ns), np.array(times), np.array(mems), np.array(time_stds), np.array(mem_stds)

def get_means(algo_data):
    """Backward compatibility - returns only means"""
    ns, times, mems, _, _ = get_means_and_stds(algo_data)
    return ns, times, mems

def plot_results(data, dist_name):
    save_dir = ensure_dirs(dist_name)
    print(f"  Generating plots in {save_dir}...")
    
    # Get friendly display name
    display_name = DIST_DISPLAY_NAMES.get(dist_name, dist_name)

    # 1. INDIVIDUAL ALGORITHM PLOTS (with error bars and improved styling)
    for name, algo_data in data.items():
        if name == "SciPy": continue 

        ns, means_t, means_m, stds_t, stds_m = get_means_and_stds(algo_data)
        ns_scipy, scipy_t, scipy_m, scipy_stds_t, scipy_stds_m = get_means_and_stds(data['SciPy'])

        # --- Time Plot ---
        plt.figure(figsize=(10, 6))
        
        # Plot raw runs with reduced opacity
        for n in ns:
            runs = algo_data[n]['time']
            lbl = 'Individual Runs (per N)' if n == ns[0] else ""
            plt.scatter([n]*len(runs), runs, color='blue', alpha=0.15, s=10, label=lbl)
        
        # Plot mean with error bars
        plt.errorbar(ns, means_t, yerr=stds_t, fmt='b-o', linewidth=2, 
                    markersize=4, capsize=3, label=f'{name} Mean ± Std Dev')
        plt.errorbar(ns_scipy, scipy_t, yerr=scipy_stds_t, fmt='g--^', linewidth=1.5,
                    markersize=4, capsize=3, label='SciPy Reference ± Std Dev')
        
        # Theoretical curve
        theoretical = ns * np.log2(ns)
        if len(theoretical) > 0 and theoretical[-1] > 0:
            scale = means_t[-1] / theoretical[-1] if means_t[-1] > 0 else 1
            plt.plot(ns, theoretical * scale, 'r:', linewidth=1.5, label='Theoretical O(N log N)')

        plt.xlabel('Input Size N (Number of Points)', fontsize=11)
        plt.ylabel('Time (seconds)', fontsize=11)
        plt.title(f'Empirical Time Complexity vs Theoretical\n{name}, {display_name} Distribution', fontsize=12)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name.replace(' ','_')}_Time.png"), dpi=150)
        plt.close()

        # --- Memory Plot ---
        plt.figure(figsize=(10, 6))
        
        # Plot raw runs with reduced opacity
        for n in ns:
            runs = algo_data[n]['mem']
            lbl = 'Individual Runs (per N)' if n == ns[0] else ""
            plt.scatter([n]*len(runs), runs, color='orange', alpha=0.15, s=10, label=lbl)
        
        # Plot mean with error bars
        plt.errorbar(ns, means_m, yerr=stds_m, fmt='o-', color='orange', linewidth=2,
                    markersize=4, capsize=3, label=f'{name} Mean ± Std Dev')
        plt.errorbar(ns_scipy, scipy_m, yerr=scipy_stds_m, fmt='g--^', linewidth=1.5,
                    markersize=4, capsize=3, label='SciPy Reference ± Std Dev')

        # Theoretical curve
        theoretical_mem = ns
        if len(theoretical_mem) > 0 and theoretical_mem[-1] > 0:
            scale_mem = means_m[-1] / theoretical_mem[-1] if means_m[-1] > 0 else 1
            plt.plot(ns, theoretical_mem * scale_mem, 'r:', linewidth=1.5, label='Theoretical O(N)')

        plt.xlabel('Input Size N (Number of Points)', fontsize=11)
        plt.ylabel('Memory (KB)', fontsize=11)
        plt.title(f'Empirical Memory Usage vs Theoretical\n{name}, {display_name} Distribution', fontsize=12)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name.replace(' ','_')}_Memory.png"), dpi=150)
        plt.close()

    # 2. COMBINED PLOTS (all algorithms on one chart)
    # --- Combined Time ---
    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        ns, means_t, _, stds_t, _ = get_means_and_stds(algo_data)
        plt.errorbar(ns, means_t, yerr=stds_t, label=name, linewidth=1.5, 
                    marker='o', markersize=3, capsize=2)
    plt.xlabel('Input Size N (Number of Points)', fontsize=11)
    plt.ylabel('Time (seconds)', fontsize=11)
    plt.title(f'Time Complexity Comparison\n{display_name} Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Combined_Time.png"), dpi=150)
    plt.close()

    # --- Combined Memory ---
    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        ns, _, means_m, _, stds_m = get_means_and_stds(algo_data)
        plt.errorbar(ns, means_m, yerr=stds_m, label=name, linewidth=1.5,
                    marker='o', markersize=3, capsize=2)
    plt.xlabel('Input Size N (Number of Points)', fontsize=11)
    plt.ylabel('Memory (KB)', fontsize=11)
    plt.title(f'Memory Usage Comparison\n{display_name} Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Combined_Memory.png"), dpi=150)
    plt.close()

    # 3. RATIO PLOTS (dimensionless)
    # --- Ratio Time ---
    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        ns, means_t, _, _, _ = get_means_and_stds(algo_data)
        theory = ns * np.log2(ns)
        theory[theory == 0] = 1 
        ratios = means_t / theory
        plt.plot(ns, ratios, label=name, linewidth=1.5, marker='o', markersize=3)
    plt.xlabel('Input Size N (Number of Points)', fontsize=11)
    plt.ylabel('Ratio: Time / (N log₂ N)  [dimensionless]', fontsize=11)
    plt.title(f'Time Complexity Ratio Test\n{display_name} Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ratio_Time.png"), dpi=150)
    plt.close()

    # --- Ratio Memory ---
    plt.figure(figsize=(12, 8))
    for name, algo_data in data.items():
        ns, _, means_m, _, _ = get_means_and_stds(algo_data)
        theory = ns 
        theory[theory == 0] = 1
        ratios = means_m / theory
        plt.plot(ns, ratios, label=name, linewidth=1.5, marker='o', markersize=3)
    plt.xlabel('Input Size N (Number of Points)', fontsize=11)
    plt.ylabel('Ratio: Memory / N  [dimensionless]', fontsize=11)
    plt.title(f'Memory Usage Ratio Test\n{display_name} Distribution', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Ratio_Memory.png"), dpi=150)
    plt.close()

def plot_grouped_bar_charts(all_results):
    """
    Create grouped bar charts across all distributions.
    X-axis: Distribution types
    Y-axis: Time or Memory
    Each group has bars for different algorithms
    """
    print("\n  Generating grouped bar charts across all distributions...")
    
    # Extract distribution names and algorithm names
    dist_names = sorted(all_results.keys())
    if not dist_names:
        print("  No data to plot grouped bar charts.")
        return
    
    # Get algorithm names from first distribution
    first_dist = all_results[dist_names[0]]
    algo_names = list(first_dist.keys())
    
    # Use the largest input size for comparison
    target_n = max(INPUT_SIZES)
    
    # Prepare data structures
    time_data = {algo: [] for algo in algo_names}
    time_err = {algo: [] for algo in algo_names}
    mem_data = {algo: [] for algo in algo_names}
    mem_err = {algo: [] for algo in algo_names}
    valid_dists = []
    
    for dist_name in dist_names:
        dist_data = all_results[dist_name]
        # Find closest N to target
        all_ns = set()
        for algo_data in dist_data.values():
            all_ns.update(algo_data.keys())
        
        if not all_ns:
            continue
            
        closest_n = min(all_ns, key=lambda x: abs(x - target_n))
        
        # Check if all algorithms have data for this N
        has_all = all(closest_n in dist_data[algo] for algo in algo_names)
        if not has_all:
            continue
            
        valid_dists.append(DIST_DISPLAY_NAMES.get(dist_name, dist_name))
        
        for algo in algo_names:
            times = dist_data[algo][closest_n]['time']
            mems = dist_data[algo][closest_n]['mem']
            time_data[algo].append(np.mean(times) if times else 0)
            time_err[algo].append(np.std(times) if times else 0)
            mem_data[algo].append(np.mean(mems) if mems else 0)
            mem_err[algo].append(np.std(mems) if mems else 0)
    
    if not valid_dists:
        print("  No valid distributions with complete data.")
        return
    
    x = np.arange(len(valid_dists))
    width = 0.8 / len(algo_names)
    
    # --- Grouped Bar Chart: TIME ---
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, algo in enumerate(algo_names):
        offset = (i - len(algo_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, time_data[algo], width, 
                     yerr=time_err[algo], capsize=3, label=algo)
        # Add value labels on bars
        ax.bar_label(bars, fmt='%.4f', fontsize=7, rotation=90, padding=3)
    
    ax.set_xlabel('Distribution Type', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Algorithm Time Comparison Across Distributions\n(N ≈ {target_n:,} points)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_dists, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_BASE_DIR, "Grouped_Time_Across_Distributions.png"), dpi=150)
    plt.close()
    print(f"    Saved: Grouped_Time_Across_Distributions.png")
    
    # --- Grouped Bar Chart: MEMORY ---
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, algo in enumerate(algo_names):
        offset = (i - len(algo_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, mem_data[algo], width,
                     yerr=mem_err[algo], capsize=3, label=algo)
        ax.bar_label(bars, fmt='%.0f', fontsize=7, rotation=90, padding=3)
    
    ax.set_xlabel('Distribution Type', fontsize=12)
    ax.set_ylabel('Memory (KB)', fontsize=12)
    ax.set_title(f'Algorithm Memory Comparison Across Distributions\n(N ≈ {target_n:,} points)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_dists, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_BASE_DIR, "Grouped_Memory_Across_Distributions.png"), dpi=150)
    plt.close()
    print(f"    Saved: Grouped_Memory_Across_Distributions.png")

def plot_single_algorithm_bars(all_results):
    """
    Create individual bar charts for each algorithm showing performance across all distributions.
    X-axis: Distribution types
    Y-axis: Time or Memory
    One chart per algorithm
    """
    print("\n  Generating individual algorithm bar charts...")
    
    dist_names = sorted(all_results.keys())
    if not dist_names:
        return
    
    first_dist = all_results[dist_names[0]]
    algo_names = list(first_dist.keys())
    
    target_n = max(INPUT_SIZES)
    
    # Prepare data per algorithm
    for algo in algo_names:
        time_vals = []
        time_errs = []
        mem_vals = []
        mem_errs = []
        valid_dists = []
        
        for dist_name in dist_names:
            dist_data = all_results[dist_name]
            if algo not in dist_data:
                continue
                
            algo_data = dist_data[algo]
            if not algo_data:
                continue
                
            closest_n = min(algo_data.keys(), key=lambda x: abs(x - target_n))
            
            times = algo_data[closest_n]['time']
            mems = algo_data[closest_n]['mem']
            
            if times:
                valid_dists.append(DIST_DISPLAY_NAMES.get(dist_name, dist_name))
                time_vals.append(np.mean(times))
                time_errs.append(np.std(times))
                mem_vals.append(np.mean(mems))
                mem_errs.append(np.std(mems))
        
        if not valid_dists:
            continue
        
        x = np.arange(len(valid_dists))
        
        # --- Time Bar Chart ---
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x, time_vals, yerr=time_errs, capsize=4, color='steelblue', alpha=0.8)
        ax.bar_label(bars, fmt='%.5f', fontsize=8, rotation=90, padding=3)
        ax.set_xlabel('Distribution Type', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(f'{algo}: Time Across All Distributions\n(N ≈ {target_n:,} points)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_dists, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        safe_algo_name = algo.replace(' ', '_')
        plt.savefig(os.path.join(PLOT_BASE_DIR, f"{safe_algo_name}_Time_All_Distributions.png"), dpi=150)
        plt.close()
        print(f"    Saved: {safe_algo_name}_Time_All_Distributions.png")
        
        # --- Memory Bar Chart ---
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(x, mem_vals, yerr=mem_errs, capsize=4, color='coral', alpha=0.8)
        ax.bar_label(bars, fmt='%.0f', fontsize=8, rotation=90, padding=3)
        ax.set_xlabel('Distribution Type', fontsize=11)
        ax.set_ylabel('Memory (KB)', fontsize=11)
        ax.set_title(f'{algo}: Memory Usage Across All Distributions\n(N ≈ {target_n:,} points)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_dists, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_BASE_DIR, f"{safe_algo_name}_Memory_All_Distributions.png"), dpi=150)
        plt.close()
        print(f"    Saved: {safe_algo_name}_Memory_All_Distributions.png")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark convex hull algorithms across multiple distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 benchmark.py                    # Run full benchmark and generate all plots
  python3 benchmark.py --bar-charts-only  # Generate only bar charts (fast if cache exists)
  python3 benchmark.py --no-save          # Run benchmark but don't save results cache
        """
    )
    parser.add_argument(
        '--bar-charts-only',
        action='store_true',
        help='Generate only cross-distribution bar charts (uses cache if available, runs benchmark if not)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save benchmark results to cache file'
    )
    parser.add_argument(
        '--results-file',
        default=RESULTS_CACHE_FILE,
        help=f'Path to results cache file (default: {RESULTS_CACHE_FILE})'
    )
    
    args = parser.parse_args()
    
    # Bar charts only mode
    if args.bar_charts_only:
        print("=== Bar Charts Only Mode ===")
        all_distribution_results = load_results(args.results_file)
        
        # If no cached results, run minimal benchmark to generate them
        if not all_distribution_results:
            print("\nNo cached results found. Running benchmark to generate data...")
            compile_cpp()
            all_distribution_results = {}
            
            for dist_name, dist_func in DISTRIBUTIONS.items():
                try:
                    dist_data = benchmark_distribution(dist_name, dist_func)
                    all_distribution_results[dist_name] = dist_data
                except Exception as e:
                    print(f"CRITICAL ERROR processing {dist_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save results for future use
            if all_distribution_results:
                save_results(all_distribution_results, args.results_file)
        
        # Generate bar charts
        if all_distribution_results:
            try:
                plot_grouped_bar_charts(all_distribution_results)
                plot_single_algorithm_bars(all_distribution_results)
                print(f"\nBar charts generated. Check the '{PLOT_BASE_DIR}' directory.")
            except Exception as e:
                print(f"ERROR generating cross-distribution plots: {e}")
                import traceback
                traceback.print_exc()
        
        exit(0)
    
    # Full benchmark mode
    compile_cpp()
    
    # Store all results for cross-distribution plots
    all_distribution_results = {}
    
    # Iterate over all distributions
    for dist_name, dist_func in DISTRIBUTIONS.items():
        try:
            # Run Benchmark
            dist_data = benchmark_distribution(dist_name, dist_func)
            # Store results
            all_distribution_results[dist_name] = dist_data
            # Generate per-distribution plots
            plot_results(dist_data, dist_name)
        except Exception as e:
            print(f"CRITICAL ERROR processing {dist_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to cache (unless --no-save was specified)
    if not args.no_save and all_distribution_results:
        save_results(all_distribution_results, args.results_file)
    
    # Generate cross-distribution comparison plots
    if all_distribution_results:
        try:
            plot_grouped_bar_charts(all_distribution_results)
            plot_single_algorithm_bars(all_distribution_results)
        except Exception as e:
            print(f"ERROR generating cross-distribution plots: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nAll benchmarks complete. Check the '{PLOT_BASE_DIR}' directory.")
    print(f"  - Per-distribution plots in subdirectories")
    print(f"  - Cross-distribution grouped bar charts in main directory")
    print(f"  - Individual algorithm bar charts in main directory")
    if not args.no_save:
        print(f"  - Results cached in '{args.results_file}' for later use")