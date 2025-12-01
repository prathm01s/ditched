"""
Graph generation script for performance analysis.

This script generates performance graphs from benchmark results, including
time vs input size, memory vs input size, and empirical vs theoretical comparisons.
"""

import sys
import os
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results_csv(filename: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from CSV file.
    
    Args:
        filename: Path to CSV file
        
    Returns:
        List of result dictionaries
    """
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['input_size'] = int(row['input_size'])
            row['trial'] = int(row['trial'])
            if row.get('time_ms') and row['time_ms'] != '':
                row['time_ms'] = float(row['time_ms'])
            else:
                row['time_ms'] = None
            if row.get('memory_mb') and row['memory_mb'] != '':
                row['memory_mb'] = float(row['memory_mb'])
            else:
                row['memory_mb'] = None
            if row.get('faces') and row['faces'] != '':
                row['faces'] = int(row['faces'])
            else:
                row['faces'] = None
            if row.get('vertices') and row['vertices'] != '':
                row['vertices'] = int(row['vertices'])
            else:
                row['vertices'] = None
            results.append(row)
    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Aggregate results by algorithm and input size.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary: {algorithm: {input_size: {metric: average_value}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for result in results:
        if not result.get('success', False):
            continue
        
        algo = result['algorithm']
        size = result['input_size']
        
        if result.get('time_ms'):
            aggregated[algo][size]['time_ms'].append(result['time_ms'])
        if result.get('memory_mb'):
            aggregated[algo][size]['memory_mb'].append(result['memory_mb'])
        if result.get('faces'):
            aggregated[algo][size]['faces'].append(result['faces'])
        if result.get('vertices'):
            aggregated[algo][size]['vertices'].append(result['vertices'])
    
    # Compute averages
    final = {}
    for algo in aggregated:
        final[algo] = {}
        for size in aggregated[algo]:
            final[algo][size] = {}
            for metric in aggregated[algo][size]:
                values = aggregated[algo][size][metric]
                if values:
                    final[algo][size][metric] = np.mean(values)
    
    return final


def generate_time_graph(aggregated: Dict, output_path: str):
    """
    Generate time vs input size graph with theoretical complexity overlays.
    
    Args:
        aggregated: Aggregated results dictionary
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['quickhull', 'jarvis', 'incremental']
    colors = {'quickhull': 'blue', 'jarvis': 'green', 'incremental': 'red'}
    labels = {'quickhull': 'QuickHull', 'jarvis': 'Jarvis March', 'incremental': 'Incremental'}
    
    for algo in algorithms:
        if algo not in aggregated:
            continue
        
        sizes = sorted(aggregated[algo].keys())
        times = [aggregated[algo][s].get('time_ms', 0) for s in sizes]
        
        ax.plot(sizes, times, 'o-', label=labels[algo], color=colors[algo], linewidth=2, markersize=6)
    
    # Add theoretical complexity reference lines
    sizes = sorted(set(s for algo in aggregated for s in aggregated[algo].keys()))
    if sizes:
        min_size, max_size = min(sizes), max(sizes)
        n_range = np.logspace(np.log10(min_size), np.log10(max_size), 100)
        
        # O(n log n) reference (scaled to fit)
        if 'quickhull' in aggregated and sizes:
            ref_times = [aggregated['quickhull'][s].get('time_ms', 0) for s in sizes if s in aggregated['quickhull']]
            if ref_times:
                scale = max(ref_times) / (max_size * np.log(max_size))
                ax.plot(n_range, scale * n_range * np.log(n_range), '--', 
                       color='lightblue', alpha=0.5, label='O(n log n) reference')
        
        # O(n²) reference (scaled to fit)
        if 'incremental' in aggregated and sizes:
            ref_times = [aggregated['incremental'][s].get('time_ms', 0) for s in sizes if s in aggregated['incremental']]
            if ref_times:
                scale = max(ref_times) / (max_size ** 2)
                ax.plot(n_range, scale * n_range ** 2, '--', 
                       color='lightcoral', alpha=0.5, label='O(n²) reference')
    
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Algorithm Performance: Time vs Input Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_memory_graph(aggregated: Dict, output_path: str):
    """
    Generate memory vs input size graph.
    
    Args:
        aggregated: Aggregated results dictionary
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['quickhull', 'jarvis', 'incremental']
    colors = {'quickhull': 'blue', 'jarvis': 'green', 'incremental': 'red'}
    labels = {'quickhull': 'QuickHull', 'jarvis': 'Jarvis March', 'incremental': 'Incremental'}
    
    for algo in algorithms:
        if algo not in aggregated:
            continue
        
        sizes = sorted(aggregated[algo].keys())
        memories = [aggregated[algo][s].get('memory_mb', 0) for s in sizes]
        
        ax.plot(sizes, memories, 'o-', label=labels[algo], color=colors[algo], linewidth=2, markersize=6)
    
    ax.set_xlabel('Input Size (n)', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Algorithm Performance: Memory vs Input Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_empirical_vs_theoretical(aggregated: Dict, output_path: str):
    """
    Generate empirical vs theoretical complexity comparison graph.
    
    Args:
        aggregated: Aggregated results dictionary
        output_path: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algorithms = ['quickhull', 'jarvis', 'incremental']
    colors = {'quickhull': 'blue', 'jarvis': 'green', 'incremental': 'red'}
    labels = {'quickhull': 'QuickHull', 'jarvis': 'Jarvis March', 'incremental': 'Incremental'}
    
    # Get all sizes
    sizes = sorted(set(s for algo in aggregated for s in aggregated[algo].keys()))
    if not sizes:
        plt.close()
        return
    
    min_size, max_size = min(sizes), max(sizes)
    n_range = np.logspace(np.log10(min_size), np.log10(max_size), 100)
    
    # Plot 1: QuickHull O(n log n) comparison
    if 'quickhull' in aggregated:
        qh_sizes = sorted(aggregated['quickhull'].keys())
        qh_times = [aggregated['quickhull'][s].get('time_ms', 0) for s in qh_sizes]
        
        ax1.plot(qh_sizes, qh_times, 'o-', label='QuickHull (Empirical)', 
                color=colors['quickhull'], linewidth=2, markersize=6)
        
        # Fit O(n log n) curve
        if qh_times:
            # Scale reference curve to match data
            scale = max(qh_times) / (max_size * np.log(max_size))
            ax1.plot(n_range, scale * n_range * np.log(n_range), '--', 
                   color='lightblue', linewidth=2, label='O(n log n) Theoretical', alpha=0.7)
        
        ax1.set_xlabel('Input Size (n)', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('QuickHull: Empirical vs O(n log n)', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Incremental O(n²) comparison
    if 'incremental' in aggregated:
        inc_sizes = sorted(aggregated['incremental'].keys())
        inc_times = [aggregated['incremental'][s].get('time_ms', 0) for s in inc_sizes]
        
        ax2.plot(inc_sizes, inc_times, 'o-', label='Incremental (Empirical)', 
                color=colors['incremental'], linewidth=2, markersize=6)
        
        # Fit O(n²) curve
        if inc_times:
            # Scale reference curve to match data
            scale = max(inc_times) / (max_size ** 2)
            ax2.plot(n_range, scale * n_range ** 2, '--', 
                   color='lightcoral', linewidth=2, label='O(n²) Theoretical', alpha=0.7)
        
        ax2.set_xlabel('Input Size (n)', fontsize=12)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Incremental: Empirical vs O(n²)', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_table(aggregated: Dict, output_path: str):
    """
    Generate comparison table as CSV.
    
    Args:
        aggregated: Aggregated results dictionary
        output_path: Output file path
    """
    sizes = sorted(set(s for algo in aggregated for s in aggregated[algo].keys()))
    algorithms = ['quickhull', 'jarvis', 'incremental']
    labels = {'quickhull': 'QuickHull', 'jarvis': 'Jarvis March', 'incremental': 'Incremental'}
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input Size'] + [f'{labels[algo]} Time (ms)' for algo in algorithms] + 
                       [f'{labels[algo]} Memory (MB)' for algo in algorithms])
        
        for size in sizes:
            row = [size]
            for algo in algorithms:
                if algo in aggregated and size in aggregated[algo]:
                    row.append(f"{aggregated[algo][size].get('time_ms', 0):.2f}")
                else:
                    row.append('N/A')
            for algo in algorithms:
                if algo in aggregated and size in aggregated[algo]:
                    row.append(f"{aggregated[algo][size].get('memory_mb', 0):.2f}")
                else:
                    row.append('N/A')
            writer.writerow(row)


def main():
    """Main entry point for graph generation."""
    parser = argparse.ArgumentParser(description='Generate performance graphs from benchmark results')
    parser.add_argument('--input', type=str, default='results/benchmark_results.csv',
                       help='Input CSV file with benchmark results')
    parser.add_argument('--output', type=str, default='analysis/graphs',
                       help='Output directory for graphs')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results_csv(args.input)
    print(f"Loaded {len(results)} results")
    
    # Aggregate results
    print("Aggregating results...")
    aggregated = aggregate_results(results)
    
    # Generate graphs
    print("Generating graphs...")
    generate_time_graph(aggregated, os.path.join(args.output, 'time_vs_input_size.png'))
    print("  [OK] Generated time_vs_input_size.png")
    
    generate_memory_graph(aggregated, os.path.join(args.output, 'memory_vs_input_size.png'))
    print("  [OK] Generated memory_vs_input_size.png")
    
    generate_comparison_table(aggregated, os.path.join(args.output, 'comparison_table.csv'))
    print("  [OK] Generated comparison_table.csv")
    
    generate_empirical_vs_theoretical(aggregated, os.path.join(args.output, 'empirical_vs_theoretical.png'))
    print("  [OK] Generated empirical_vs_theoretical.png")
    
    print(f"\nGraphs saved to {args.output}")


if __name__ == '__main__':
    main()

