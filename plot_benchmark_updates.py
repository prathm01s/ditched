import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

# --- Configuration ---
RESULTS_FILE = "benchmark_results_latest1.pkl"
PLOT_DIR = "plots_latest1/updated_visuals"
TARGET_N_STANDARD = 700000
TARGET_N_JARVIS = 25000

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

def get_closest_n_data(algo_data, target_n):
    """Finds the N in the dataset closest to the target N."""
    if not algo_data:
        return None, (0, 0), (0, 0)
    
    available_ns = list(algo_data.keys())
    closest_n = min(available_ns, key=lambda x: abs(x - target_n))
    
    # Sanity check: if closest N is too far off (e.g. < 50% of target), return empty
    if closest_n < target_n * 0.5:
        return None, (0, 0), (0, 0)

    metrics = algo_data[closest_n]
    
    # Calculate Time Stats
    if metrics['time']:
        t_mean = np.mean(metrics['time'])
        t_std = np.std(metrics['time'])
    else:
        t_mean, t_std = 0, 0
        
    # Calculate Memory Stats
    if metrics['mem']:
        m_mean = np.mean(metrics['mem'])
        m_std = np.std(metrics['mem'])
    else:
        m_mean, m_std = 0, 0
        
    return closest_n, (t_mean, t_std), (m_mean, m_std)

def plot_grouped_memory(all_results):
    print("Generating Grouped Memory Chart (All Algorithms incl. Jarvis) [Log Scale]...")
    
    algos = ["Graham Scan", "Monotone Chain", "Optimized Monotone", "QuickHull", "SciPy", "Jarvis March"]
    dist_names = sorted(all_results.keys())
    
    valid_dists = []
    plot_data = {algo: {'means': [], 'stds': []} for algo in algos}
    
    for dist in dist_names:
        has_data = False
        temp_means = {}
        temp_stds = {}
        
        for algo in algos:
            if algo not in all_results[dist]: 
                temp_means[algo] = 0; temp_stds[algo] = 0
                continue
            
            # DYNAMIC N SELECTION: 25k for Jarvis, 700k for others
            target_n = TARGET_N_JARVIS if algo == "Jarvis March" else TARGET_N_STANDARD
            
            n, _, (m_mean, m_std) = get_closest_n_data(all_results[dist][algo], target_n)
            
            if n:
                has_data = True
                temp_means[algo] = m_mean
                temp_stds[algo] = m_std
            else:
                temp_means[algo] = 0; temp_stds[algo] = 0
        
        if has_data:
            valid_dists.append(DIST_DISPLAY_NAMES.get(dist, dist))
            for algo in algos:
                plot_data[algo]['means'].append(temp_means[algo])
                plot_data[algo]['stds'].append(temp_stds[algo])

    # Plotting
    x = np.arange(len(valid_dists))
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(18, 9))
    
    for i, algo in enumerate(algos):
        offset = (i - len(algos)/2 + 0.5) * width
        means = plot_data[algo]['means']
        stds = plot_data[algo]['stds']
        
        # FIX: Ensure error bars are never negative.
        # If mean is 0, lower error is 0. 
        # If mean > 0, we clamp lower error so the bar doesn't go below 0.1 (log scale safety)
        lower_err = [0 if m <= 0.1 else min(m - 0.1, s) for m, s in zip(means, stds)]
        upper_err = stds
        errs = [lower_err, upper_err]

        ax.bar(x + offset, means, width, yerr=errs, 
               label=algo, capsize=2, color=COLORS[i % len(COLORS)], alpha=0.9)

    ax.set_yscale('log')
    ax.set_ylabel('Memory Usage (KB) [Log Scale]', fontsize=12, fontweight='bold')
    
    # Update title to reflect the mixed N values
    ax.set_title(f'Memory Comparison (N≈{TARGET_N_STANDARD//1000}k, Jarvis N≈{TARGET_N_JARVIS//1000}k)', 
                 fontsize=16, fontweight='bold')
                 
    ax.set_xticks(x)
    ax.set_xticklabels(valid_dists, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Grouped_Memory_All_Algos_Log.png"), dpi=200)
    plt.close()

def plot_individual_algo(all_results, algo_name, metric_key, target_n, ylabel):
    print(f"Generating Individual {metric_key.capitalize()} Chart for {algo_name} [Log Scale]...")
    
    dist_names = sorted(all_results.keys())
    means = []
    stds = []
    labels = []
    
    for dist in dist_names:
        if algo_name not in all_results[dist]: continue
        
        n, (t_mean, t_std), (m_mean, m_std) = get_closest_n_data(all_results[dist][algo_name], target_n)
        
        if n:
            labels.append(DIST_DISPLAY_NAMES.get(dist, dist))
            if metric_key == 'time':
                means.append(t_mean)
                stds.append(t_std)
            else:
                means.append(m_mean)
                stds.append(m_std)

    if not means:
        print(f"  No data found for {algo_name}")
        return

    # Dynamic Coloring
    norm = plt.Normalize(min(means), max(means))
    cmap = plt.cm.viridis if metric_key == 'time' else plt.cm.magma
    bar_colors = cmap(norm(means))

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(labels, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.85)
    
    ax.set_yscale('log')
    ax.set_ylabel(f"{ylabel} [Log Scale]", fontsize=12)
    ax.set_title(f'{algo_name}: {metric_key.capitalize()} (N ≈ {target_n:,})', fontsize=15, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    
    filename = f"{algo_name.replace(' ', '_')}_{metric_key.capitalize()}_Log.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=200)
    plt.close()

def main():
    ensure_dir()
    results = load_results(RESULTS_FILE)
    
    # 1. Grouped Memory (All Algos + Jarvis, Log Scale)
    plot_grouped_memory(results)
    
    # 2. Individual Time Plot for Jarvis (N=25k)
    plot_individual_algo(results, "Jarvis March", "time", TARGET_N_JARVIS, "Time (s)")
    
    # 3. Individual Memory Plots for All Algos
    algos = ["Graham Scan", "Monotone Chain", "Optimized Monotone", "QuickHull", "SciPy", "Jarvis March"]
    for algo in algos:
        n = TARGET_N_JARVIS if algo == "Jarvis March" else TARGET_N_STANDARD
        plot_individual_algo(results, algo, "mem", n, "Memory (KB)")

    print(f"\nAll plots generated in directory: {PLOT_DIR}")

if __name__ == "__main__":
    main()