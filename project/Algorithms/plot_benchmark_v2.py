import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# --- Configuration ---
RESULTS_FILE = "benchmark_results_latest1.pkl"
PLOT_DIR = "plots_latest1/log_scale_only"  # New clean directory

# Target N values
TARGET_N_STANDARD = 700000
TARGET_N_JARVIS = 100000 

# Aesthetic Config
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab10.colors 

DIST_DISPLAY_NAMES = {
    "1_uniform_square": "Uniform Square",
    "2_gaussian_blob": "Gaussian Blob",
    "3_multi_cluster_200_instance_1": "Multi-Cluster (1)",
    "3_multi_cluster_200_instance_2": "Multi-Cluster (2)",
    "3_multi_cluster_200_instance_3": "Multi-Cluster (3)",
    "3_multi_cluster_200_instance_4": "Multi-Cluster (4)",
    "3_multi_cluster_200_instance_5": "Multi-Cluster (5)",
    "4_uniform_disk": "Uniform Disk",
    "5_circle_perimeter": "Circle Perimeter",
    "6_rectangle_perimeter": "Rect Perimeter",
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
    "17_poisson_process": "Poisson",
    "18_onion": "Onion"
}

def ensure_dir():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

def load_results(filename):
    print(f"Loading results from {filename}...")
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        exit(1)

def get_best_n_data(algo_data, target_n):
    """
    Finds the N closest to target_n. 
    If target_n is far away, it falls back to the largest available N.
    """
    if not algo_data:
        return None, (0, 0), (0, 0)
    
    available_ns = sorted(list(algo_data.keys()))
    
    # 1. Try finding closest N
    closest_n = min(available_ns, key=lambda x: abs(x - target_n))
    
    # 2. Heuristic: If match is decent (within 50% margin), take it.
    # Otherwise, fallback to max N available.
    if abs(closest_n - target_n) < target_n * 0.5:
        selected_n = closest_n
    else:
        selected_n = max(available_ns)

    metrics = algo_data[selected_n]
    
    # Calculate Stats
    t_mean = np.mean(metrics['time']) if metrics['time'] else 0
    t_std = np.std(metrics['time']) if metrics['time'] else 0
    
    m_mean = np.mean(metrics['mem']) if metrics['mem'] else 0
    m_std = np.std(metrics['mem']) if metrics['mem'] else 0
        
    return selected_n, (t_mean, t_std), (m_mean, m_std)

def get_safe_log_errs(means, stds):
    """
    Calculates asymmetric error bars safe for log scale.
    Clips the lower error bar so it doesn't cross zero.
    """
    lowers = []
    uppers = []
    for m, s in zip(means, stds):
        if m <= 0:
            lowers.append(0)
            uppers.append(0)
            continue
            
        # Cap lower error so the bar stops at 1% of the value at worst
        safe_floor = m * 0.01 
        max_valid_s = m - safe_floor
        
        effective_lower_s = min(s, max_valid_s)
        if effective_lower_s < 0: effective_lower_s = 0
        
        lowers.append(effective_lower_s)
        uppers.append(s)
        
    return [lowers, uppers]

def plot_grouped_metric(all_results, metric_key):
    if metric_key == 'time':
        ylabel = "Time (s)"
        title_metric = "Time"
        save_name = "Grouped_Time_Log.png"
    else:
        ylabel = "Memory (KB)"
        title_metric = "Memory"
        save_name = "Grouped_Memory_Log.png"

    print(f"Generating Grouped {title_metric} Chart [Log Only]...")
    
    algos = ["Graham Scan", "Monotone Chain", "Optimized Monotone", "QuickHull", "SciPy", "Jarvis March"]
    dist_names = sorted(all_results.keys())
    
    valid_dists = []
    plot_data = {algo: {'means': [], 'stds': []} for algo in algos}
    
    # Collect Data
    for dist in dist_names:
        has_data = False
        temp_means = {}
        temp_stds = {}
        
        for algo in algos:
            if algo not in all_results[dist]: 
                temp_means[algo] = 0; temp_stds[algo] = 0
                continue
            
            target_n = TARGET_N_JARVIS if algo == "Jarvis March" else TARGET_N_STANDARD
            n, (t_m, t_s), (m_m, m_s) = get_best_n_data(all_results[dist][algo], target_n)
            
            val_mean = t_m if metric_key == 'time' else m_m
            val_std = t_s if metric_key == 'time' else m_s
            
            if n:
                has_data = True
                temp_means[algo] = val_mean
                temp_stds[algo] = val_std
            else:
                temp_means[algo] = 0; temp_stds[algo] = 0
        
        if has_data:
            valid_dists.append(DIST_DISPLAY_NAMES.get(dist, dist))
            for algo in algos:
                plot_data[algo]['means'].append(temp_means[algo])
                plot_data[algo]['stds'].append(temp_stds[algo])

    if not valid_dists:
        print(f"  No data found for grouped {metric_key}.")
        return

    # Plotting
    x = np.arange(len(valid_dists))
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(18, 9))
    
    for i, algo in enumerate(algos):
        offset = (i - len(algos)/2 + 0.5) * width
        means = plot_data[algo]['means']
        stds = plot_data[algo]['stds']
        
        errs = get_safe_log_errs(means, stds)
        
        ax.bar(x + offset, means, width, yerr=errs, 
               label=algo, capsize=2, color=COLORS[i % len(COLORS)], alpha=0.9,
               error_kw={'elinewidth': 1, 'capthick': 1})

    # STRICTLY LOG SCALE
    ax.set_yscale('log')
    
    ax.set_ylabel(f"{ylabel} [Log Scale]", fontsize=12, fontweight='bold')
    ax.set_title(f'{title_metric} Comparison (Log Scale)\n(Standard N≈{TARGET_N_STANDARD//1000}k, Jarvis N≈{TARGET_N_JARVIS//1000}k)', 
                 fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(valid_dists, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, save_name)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved {path}")

def plot_individual_algo(all_results, algo_name, metric_key):
    target_n = TARGET_N_JARVIS if algo_name == "Jarvis March" else TARGET_N_STANDARD
    if metric_key == 'time':
        ylabel = "Time (s)"
        title_metric = "Time"
    else:
        ylabel = "Memory (KB)"
        title_metric = "Memory"

    print(f"Generating {algo_name} {title_metric} Chart [Log Only]...")
    
    dist_names = sorted(all_results.keys())
    means = []
    stds = []
    labels = []
    actual_ns = []
    
    for dist in dist_names:
        if algo_name not in all_results[dist]: continue
        
        n, (t_m, t_s), (m_m, m_s) = get_best_n_data(all_results[dist][algo_name], target_n)
        
        if n:
            labels.append(DIST_DISPLAY_NAMES.get(dist, dist))
            actual_ns.append(n)
            if metric_key == 'time':
                means.append(t_m); stds.append(t_s)
            else:
                means.append(m_m); stds.append(m_s)

    if not means:
        print(f"  Skipping {algo_name} (No data found).")
        return

    # Dynamic Coloring
    norm = plt.Normalize(min(means) if min(means) > 0 else 0, max(means))
    cmap = plt.cm.viridis if metric_key == 'time' else plt.cm.magma
    bar_colors = cmap(norm(means))

    fig, ax = plt.subplots(figsize=(16, 7))
    
    errs = get_safe_log_errs(means, stds)
    
    ax.bar(labels, means, yerr=errs, capsize=4, color=bar_colors, alpha=0.85)
    
    # STRICTLY LOG SCALE
    ax.set_yscale('log')
    
    # Calculate average N for title
    avg_n = int(np.mean(actual_ns)) if actual_ns else 0
    
    ax.set_ylabel(f"{ylabel} [Log Scale]", fontsize=12)
    ax.set_title(f'{algo_name}: {title_metric} (Log Scale, N ≈ {avg_n:,})', fontsize=15, fontweight='bold')
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    
    filename = f"{algo_name.replace(' ', '_')}_{title_metric}_Log.png"
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved {path}")

def main():
    ensure_dir()
    results = load_results(RESULTS_FILE)
    
    # 1. Grouped Charts (Time & Memory)
    plot_grouped_metric(results, 'time')
    plot_grouped_metric(results, 'mem')
    
    # 2. Individual Charts (All Algos, Time & Memory)
    algos = ["Graham Scan", "Monotone Chain", "Optimized Monotone", "QuickHull", "SciPy", "Jarvis March"]
    
    print("\n--- Generating Individual Plots (Log Scale Only) ---")
    for algo in algos:
        plot_individual_algo(results, algo, 'time')
        plot_individual_algo(results, algo, 'mem')

    print(f"\nAll log-scale plots generated in directory: {PLOT_DIR}")

if __name__ == "__main__":
    main()