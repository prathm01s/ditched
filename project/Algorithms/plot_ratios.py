import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# --- Configuration ---
RESULTS_FILE = "benchmark_results_latest1.pkl"
PLOT_DIR = "plots_latest1/asymptotic_checks"

# Aesthetic Config
plt.style.use('seaborn-v0_8-whitegrid')
# High contrast colors
COLORS = plt.cm.tab10.colors 

def ensure_dirs():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    if not os.path.exists(os.path.join(PLOT_DIR, "per_distribution")):
        os.makedirs(os.path.join(PLOT_DIR, "per_distribution"))
    if not os.path.exists(os.path.join(PLOT_DIR, "per_algorithm")):
        os.makedirs(os.path.join(PLOT_DIR, "per_algorithm"))

def load_results(filename):
    print(f"Loading results from {filename}...")
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        exit(1)

def get_mean_values(algo_data):
    """
    Extracts sorted Ns and corresponding mean Times and Memories.
    """
    if not algo_data:
        return [], [], []
    
    ns = sorted(algo_data.keys())
    times = []
    mems = []
    
    for n in ns:
        metrics = algo_data[n]
        t = np.mean(metrics['time']) if metrics['time'] else 0
        m = np.mean(metrics['mem']) if metrics['mem'] else 0
        times.append(t)
        mems.append(m)
        
    return np.array(ns), np.array(times), np.array(mems)

def plot_per_distribution(all_results):
    print("Generating Asymptotic Plots per Distribution...")
    
    dist_names = sorted(all_results.keys())
    
    for dist in dist_names:
        # Prepare figure with 2 subplots: Time Ratio and Memory Ratio
        fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(18, 7))
        
        has_data = False
        
        for i, algo in enumerate(sorted(all_results[dist].keys())):
            ns, times, mems = get_mean_values(all_results[dist][algo])
            if len(ns) < 2: continue # Need at least 2 points for a line
            has_data = True
            
            # --- Time Asymptotic Ratio: Time / (N * logN) ---
            # We add a small epsilon to N to avoid div by zero if N=0 (unlikely)
            # Theoretical Complexity C(N) = N log2(N)
            complexity = ns * np.log2(ns)
            time_ratios = times / complexity
            
            ax_time.plot(ns, time_ratios, marker='o', markersize=4, 
                         label=algo, color=COLORS[i % len(COLORS)], alpha=0.8)
            
            # --- Memory Asymptotic Ratio: Memory / N ---
            mem_ratios = mems / ns
            ax_mem.plot(ns, mem_ratios, marker='s', markersize=4, 
                        label=algo, color=COLORS[i % len(COLORS)], alpha=0.8)

        if not has_data:
            plt.close()
            continue
            
        # Formatting Time Plot
        ax_time.set_title(f"Time Asymptotic Constant ($T / N \log N$)\n{dist}", fontsize=12, fontweight='bold')
        ax_time.set_xlabel("N")
        ax_time.set_ylabel("Ratio (Lower is Better)")
        ax_time.legend()
        ax_time.grid(True, linestyle='--', alpha=0.5)
        
        # If Jarvis is present (O(N^2) or O(NH)), the ratio might blow up. 
        # Use log scale on Y if ranges are huge.
        ax_time.set_yscale('log')

        # Formatting Memory Plot
        ax_mem.set_title(f"Memory Asymptotic Constant ($M / N$)\n{dist}", fontsize=12, fontweight='bold')
        ax_mem.set_xlabel("N")
        ax_mem.set_ylabel("KB per Point")
        ax_mem.legend()
        ax_mem.grid(True, linestyle='--', alpha=0.5)
        # Memory is usually stable, linear scale is fine, but log is safer for outliers
        ax_mem.set_yscale('log')

        plt.tight_layout()
        filename = f"{dist}_Asymptotic.png"
        plt.savefig(os.path.join(PLOT_DIR, "per_distribution", filename), dpi=150)
        plt.close()

def plot_per_algorithm(all_results):
    print("Generating Asymptotic Plots per Algorithm...")
    
    # 1. Identify all unique algorithms
    all_algos = set()
    for d in all_results.values():
        all_algos.update(d.keys())
        
    for algo in sorted(all_algos):
        fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(18, 7))
        has_data = False
        
        # Plot each distribution as a line
        dist_list = sorted(all_results.keys())
        for i, dist in enumerate(dist_list):
            if algo not in all_results[dist]: continue
            
            ns, times, mems = get_mean_values(all_results[dist][algo])
            if len(ns) < 2: continue
            has_data = True
            
            # Time Ratio
            complexity = ns * np.log2(ns)
            time_ratios = times / complexity
            ax_time.plot(ns, time_ratios, marker='.', markersize=3, 
                         label=dist, alpha=0.7) # Let matplotlib cycle colors
            
            # Memory Ratio
            mem_ratios = mems / ns
            ax_mem.plot(ns, mem_ratios, marker='.', markersize=3, 
                        label=dist, alpha=0.7)

        if not has_data:
            plt.close()
            continue

        # Formatting
        ax_time.set_title(f"{algo}: Time Efficiency ($T / N \log N$)", fontsize=12, fontweight='bold')
        ax_time.set_xlabel("N")
        ax_time.set_ylabel("Ratio")
        ax_time.set_yscale('log')
        # Only show legend if not too cluttered (optional)
        if len(dist_list) < 15: ax_time.legend(fontsize=8)

        ax_mem.set_title(f"{algo}: Memory Efficiency ($M / N$)", fontsize=12, fontweight='bold')
        ax_mem.set_xlabel("N")
        ax_mem.set_ylabel("KB / Point")
        ax_mem.set_yscale('log')
        
        plt.tight_layout()
        filename = f"{algo.replace(' ', '_')}_Asymptotic_All_Dists.png"
        plt.savefig(os.path.join(PLOT_DIR, "per_algorithm", filename), dpi=150)
        plt.close()

def main():
    ensure_dirs()
    results = load_results(RESULTS_FILE)
    
    print("Note: Using Full Range of N to visualize Asymptotic Trends.")
    print("      (Single point N=100k cannot show asymptotic behavior)")
    
    plot_per_distribution(results)
    plot_per_algorithm(results)
    
    print(f"\nAsymptotic ratio plots saved in: {PLOT_DIR}")

if __name__ == "__main__":
    main()