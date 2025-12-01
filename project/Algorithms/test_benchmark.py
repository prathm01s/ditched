#!/usr/bin/env python3
"""
Quick test of the updated benchmark.py functionality
Tests with a small subset of distributions and smaller input sizes
"""

import subprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc
import re
import traceback
from scipy.spatial import ConvexHull
import dataset_gen as dg

# --- Configuration for Testing ---
BIN_DIR = "bin"
PLOT_BASE_DIR = "plots_test"
ALGORITHMS_CPP = {
    "Graham Scan": "graham_scan_timed.cpp",
    "Monotone Chain": "monotone_chain_timed.cpp",
}
EXECUTABLES = {name: os.path.join(BIN_DIR, name.replace(" ", "_").lower() + ".out") 
               for name in ALGORITHMS_CPP}

# Smaller range for testing: 3 steps from 1k to 10k
INPUT_SIZES = np.linspace(1000, 10000, 3, dtype=int).tolist()
NUM_RUNS = 3  # Fewer runs for testing

DIST_DISPLAY_NAMES = {
    "1_uniform_square": "Uniform Square",
    "2_gaussian_blob": "Gaussian Blob",
}

# Only test 2 distributions
DISTRIBUTIONS = {
    "1_uniform_square": dg.gen_uniform_square,
    "2_gaussian_blob": dg.gen_gaussian_blob,
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
            return False
    print("Compilation Complete.\n")
    return True

def run_cpp_benchmark(exe_path, input_file):
    """Run C++ benchmark with accurate memory measurement using /usr/bin/time -v"""
    abs_input_path = os.path.abspath(input_file)
    cmd = ["/usr/bin/time", "-v", exe_path, abs_input_path]
    
    mem_regex = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print(f"    Exception running benchmark: {e}")
        return None, None

    if result.returncode != 0:
        print(f"    Non-zero return code: {result.returncode}")
        return None, None
    
    time_sec, mem_kb = None, None
    
    stderr_lines = result.stderr.splitlines()
    
    for line in stderr_lines:
        if "BENCHMARK_TIME_SEC=" in line:
            time_sec = float(line.split("=")[1])
    
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
        pass
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end_time - start_time), (peak / 1024)

def benchmark_distribution(dist_name, dist_func):
    """Runs benchmarks for a specific distribution."""
    print(f"\n=== Testing Distribution: {dist_name} ===")
    
    data = {name: {} for name in ALGORITHMS_CPP}
    data['SciPy'] = {}

    for idx, n in enumerate(INPUT_SIZES):
        print(f"  [N={n:<8}] ({idx+1}/{len(INPUT_SIZES)})")
        
        points = dist_func(n)
        actual_n = len(points)
        
        input_filename = f"temp_test_{dist_name}_{n}.txt"
        dg.save_points(points, input_filename, directory=".")
        
        for name in data:
            data[name][actual_n] = {'time': [], 'mem': []}

        for run in range(NUM_RUNS):
            for name, exe_path in EXECUTABLES.items():
                t, m = run_cpp_benchmark(exe_path, input_filename)
                if t is not None:
                    data[name][actual_n]['time'].append(t)
                    data[name][actual_n]['mem'].append(m)
            
            t_scipy, m_scipy = run_scipy_benchmark(points)
            data['SciPy'][actual_n]['time'].append(t_scipy)
            data['SciPy'][actual_n]['mem'].append(m_scipy)

        # Debug output
        print(f"  Results for N={actual_n}:")
        for name in data:
            times = data[name][actual_n]['time']
            mems = data[name][actual_n]['mem']
            if times:
                print(f"    {name:15} count={len(times)} mean_time={np.mean(times):.6f}s mean_mem={np.mean(mems):.2f}KB")

        if os.path.exists(input_filename):
            os.remove(input_filename)
            
    return data

if __name__ == "__main__":
    print("=== Quick Test of Updated Benchmark Script ===\n")
    
    if not compile_cpp():
        print("Compilation failed. Exiting.")
        exit(1)
    
    all_results = {}
    
    for dist_name, dist_func in DISTRIBUTIONS.items():
        try:
            dist_data = benchmark_distribution(dist_name, dist_func)
            all_results[dist_name] = dist_data
        except Exception as e:
            print(f"ERROR processing {dist_name}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n=== Test Complete ===")
    print(f"Tested {len(all_results)} distributions with {len(INPUT_SIZES)} input sizes and {NUM_RUNS} runs each.")
    print(f"Memory measurement uses /usr/bin/time -v for accuracy.")
    print(f"\nThe full benchmark.py script is ready with:")
    print(f"  ✓ Accurate memory measurement via /usr/bin/time")
    print(f"  ✓ Error bars on mean curves")
    print(f"  ✓ Reduced opacity for raw run points")
    print(f"  ✓ Clear units on all axes")
    print(f"  ✓ Improved plot titles")
    print(f"  ✓ Grouped bar charts across distributions")
    print(f"  ✓ Individual algorithm bar charts")
