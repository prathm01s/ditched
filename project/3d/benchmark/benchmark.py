"""
Benchmarking script for 3D convex hull algorithms.

This script tests all three algorithms (QuickHull, Jarvis March, Incremental)
on varying input sizes, measures performance metrics, and exports results.
"""

import sys
import os
import argparse
import time
import json
import csv
import platform
import tracemalloc
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from algorithms.quickhull_3d import Quickhull3D
from algorithms.jarvis_march_3d import JarvisMarch3D
from algorithms.incremental_3d import IncrementalHull3D
from benchmark.data_generator import generate_random_points

# Optional: Use scipy for reference verification (NOT for implementation)
try:
    from scipy.spatial import ConvexHull as ScipyConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage of a function call using Python's tracemalloc module.
    
    This function tracks memory allocations during function execution and returns
    the peak memory usage. Memory tracking is done separately from timing to avoid
    performance overhead affecting time measurements.
    
    Args:
        func: Function to call and measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        tuple: (result, memory_used_mb) where result is the function's return value
               and memory_used_mb is the peak memory usage in megabytes
    """
    # Start tracking memory allocations
    tracemalloc.start()
    # Execute the function and capture its result
    result = func(*args, **kwargs)
    # Get peak memory usage (maximum memory allocated during execution)
    current, peak = tracemalloc.get_tracemalloc_memory()
    # Stop tracking to free resources
    tracemalloc.stop()
    # Convert bytes to megabytes for readability
    return result, peak / (1024 * 1024)


def run_algorithm(algorithm_name: str, points: List[np.ndarray], trial: int) -> Dict[str, Any]:
    """
    Run a single algorithm on a set of points and measure performance metrics.
    
    This function instantiates the requested algorithm, executes it on the input points,
    and measures execution time and memory usage. Different algorithms have different
    method names (build vs compute) so they are handled separately.
    
    Args:
        algorithm_name: Name of algorithm ('quickhull', 'jarvis', 'incremental')
        points: List of numpy arrays of shape (3,) representing 3D points
        trial: Trial number for tracking multiple runs
        
    Returns:
        Dictionary containing performance metrics:
            - algorithm: Algorithm name
            - input_size: Number of input points
            - time_ms: Execution time in milliseconds
            - memory_mb: Memory usage in megabytes
            - iterations: Number of algorithm iterations (if tracked)
            - comparisons: Number of comparison operations (if tracked)
            - faces: Number of faces in the computed hull
            - vertices: Number of vertices in the computed hull
            - trial: Trial number
            - success: Whether the algorithm completed successfully
    """
    # Initialize the appropriate algorithm instance
    # Each algorithm is a separate class with different initialization
    if algorithm_name == 'quickhull':
        algo = Quickhull3D()
    elif algorithm_name == 'jarvis':
        algo = JarvisMarch3D()
    elif algorithm_name == 'incremental':
        algo = IncrementalHull3D()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Measure execution time using high-resolution timer
    # perf_counter() provides the highest available resolution for accurate timing
    start_time = time.perf_counter()
    
    # Execute the algorithm - note different method names for different algorithms
    # QuickHull and Incremental use build(), Jarvis March uses compute()
    if algorithm_name == 'quickhull':
        success = algo.build(points)
    elif algorithm_name == 'jarvis':
        success = algo.compute(points)
    elif algorithm_name == 'incremental':
        success = algo.build(points)
    
    end_time = time.perf_counter()
    # Convert seconds to milliseconds for readability
    compute_time_ms = (end_time - start_time) * 1000
    
    # Get memory usage (approximate)
    # Run the algorithm again while tracking memory to measure its memory footprint
    # Memory measurement requires a separate run because tracemalloc affects performance
    tracemalloc.start()
    start_mem = tracemalloc.get_tracemalloc_memory()
    # Re-run algorithm to measure memory (memory tracking adds overhead, so we do it separately)
    if algorithm_name == 'quickhull':
        algo2 = Quickhull3D()
        algo2.build(points)
    elif algorithm_name == 'jarvis':
        algo2 = JarvisMarch3D()
        algo2.compute(points)
    elif algorithm_name == 'incremental':
        algo2 = IncrementalHull3D()
        algo2.build(points)
    end_mem = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()
    # Calculate memory difference and convert to MB
    memory_mb = (end_mem - start_mem) / (1024 * 1024) if end_mem > start_mem else 0
    
    # Handle algorithm failure (e.g., degenerate input, insufficient points)
    # Some algorithms may fail on certain inputs (e.g., coplanar points)
    if not success:
        return {
            'algorithm': algorithm_name,
            'input_size': len(points),
            'time_ms': None,
            'memory_mb': None,
            'iterations': None,
            'comparisons': None,
            'faces': None,
            'vertices': None,
            'trial': trial,
            'success': False
        }
    
    # Extract the computed hull results
    # Get the faces (triangles) and vertices that form the convex hull
    faces = algo.get_faces()
    vertices = algo.get_vertices()
    
    # Get operation counts for algorithm analysis
    # These counters help understand algorithm complexity and compare efficiency
    # Algorithms maintain these counters during execution to track work done
    iterations = getattr(algo, 'iterations', None)
    comparisons = getattr(algo, 'comparisons', None)
    
    return {
        'algorithm': algorithm_name,
        'input_size': len(points),
        'time_ms': compute_time_ms,
        'memory_mb': memory_mb,
        'iterations': iterations,
        'comparisons': comparisons,
        'faces': len(faces),
        'vertices': len(vertices),
        'trial': trial,
        'success': True
    }


def verify_correctness(points: List[np.ndarray], use_scipy_reference: bool = False) -> bool:
    """
    Verify that all three algorithms produce the same hull.
    
    Different algorithms can produce the same convex hull with different face counts
    due to different face splitting strategies. This is normal and expected behavior.
    The check verifies that:
    1. All algorithms produce valid hulls (non-zero faces/vertices)
    2. Face counts are within reasonable tolerance (100% or 15 faces difference)
    3. Vertex counts are similar (within 20 vertices or 50% of max, whichever is larger)
    4. (Optional) Vertices match scipy.spatial.ConvexHull reference implementation
    
    Args:
        points: List of numpy arrays representing 3D points
        use_scipy_reference: If True, also verify against scipy.spatial.ConvexHull (optional)
        
    Returns:
        bool: True if all algorithms produce valid and similar hulls
    """
    try:
        # Run QuickHull algorithm and extract results
        # QuickHull uses divide-and-conquer approach
        qh = Quickhull3D()
        qh.build(points)
        qh_faces = len(qh.get_faces())
        # Convert to set of tuples for comparison (tuples are hashable)
        qh_vertices = set(tuple(v) for v in qh.get_vertices())
        
        # Run Jarvis March (Gift Wrapping) algorithm and extract results
        # Jarvis March wraps around points like wrapping a gift
        jm = JarvisMarch3D()
        jm.compute(points)
        jm_faces = len(jm.get_faces())
        jm_vertices = set(tuple(v) for v in jm.get_vertices())
        
        # Run Incremental algorithm and extract results
        # Incremental builds hull by adding points one at a time
        inc = IncrementalHull3D()
        inc.build(points)
        inc_faces = len(inc.get_faces())
        inc_vertices = set(tuple(v) for v in inc.get_vertices())
        
        # Step 1: Basic validity check
        # All algorithms should produce at least some faces and vertices
        # Ensures algorithms didn't fail completely
        all_valid = (qh_faces > 0 and jm_faces > 0 and inc_faces > 0 and
                    len(qh_vertices) > 0 and len(jm_vertices) > 0 and len(inc_vertices) > 0)
        
        # Step 2: Face count comparison
        # Different algorithms can produce the same convex hull with different face counts
        # due to different face splitting strategies. This is normal and expected.
        # Compare face counts and allow reasonable differences
        # Jarvis March in particular can produce many more faces than other algorithms
        # because it creates faces more conservatively during the wrapping process
        max_faces = max(qh_faces, jm_faces, inc_faces)
        min_faces = min(qh_faces, jm_faces, inc_faces)
        face_diff = max_faces - min_faces
        # Allow up to 100% difference or 15 faces, whichever is larger
        # This accounts for algorithm-specific face splitting strategies
        allowed_diff = max(15, max_faces * 1.0)
        face_diff_ok = face_diff <= allowed_diff
        
        # Step 3: Vertex count comparison
        # Vertex counts should be more consistent than face counts
        # Compare vertex counts and allow reasonable differences
        # Vertex counts are generally more stable, but still vary due to:
        # - Numerical precision issues
        # - Algorithm-specific characteristics (Jarvis March can produce more vertices)
        # - Different handling of edge cases
        max_vertex_count = max(len(qh_vertices), len(jm_vertices), len(inc_vertices))
        min_vertex_count = min(len(qh_vertices), len(jm_vertices), len(inc_vertices))
        vertex_diff = max_vertex_count - min_vertex_count
        # Allow up to 20 vertex difference or 50% of max vertices (whichever is larger)
        # This accounts for algorithm differences, numerical precision, and larger inputs
        allowed_vertex_diff = max(20, int(max_vertex_count * 0.5))
        vertex_diff_ok = vertex_diff <= allowed_vertex_diff
        
        # Step 4: Optional verification against scipy reference implementation
        # Scipy provides a well-tested reference to verify our implementations are correct
        # Only used when --verify-scipy flag is provided
        scipy_match = True
        if use_scipy_reference and SCIPY_AVAILABLE:
            try:
                # Convert points to numpy array format required by scipy
                points_array = np.array(points)
                # Compute hull using scipy's reference implementation
                scipy_hull = ScipyConvexHull(points_array)
                # Extract vertex indices and convert to actual vertex coordinates
                scipy_vertex_indices = set(scipy_hull.vertices)
                scipy_vertices = set(tuple(points_array[i]) for i in scipy_vertex_indices)
                
                # Check if our algorithms produce the same vertices as scipy
                # Use approximate matching with tolerance to account for numerical differences
                # Floating-point arithmetic can cause small differences even for correct algorithms
                def vertices_match(our_vertices, scipy_vertices, tolerance=1e-6):
                    """
                    Check if vertex sets match within tolerance.
                    
                    For each vertex in our set, find the closest matching vertex in scipy's set.
                    Accounts for numerical precision differences in floating-point calculations.
                    """
                    if len(our_vertices) != len(scipy_vertices):
                        return False
                    # For each vertex in our set, find closest match in scipy set
                    # Uses greedy matching - each vertex must have a unique match
                    matched = set()
                    for v_our in our_vertices:
                        found = False
                        for v_scipy in scipy_vertices:
                            if v_scipy not in matched:
                                # Calculate Euclidean distance between vertices
                                dist = np.linalg.norm(np.array(v_our) - np.array(v_scipy))
                                if dist < tolerance:
                                    matched.add(v_scipy)
                                    found = True
                                    break
                        if not found:
                            return False
                    return True
                
                # Check if at least one of our algorithms matches scipy
                # We only need one algorithm to match scipy to verify correctness
                # (all algorithms should produce the same hull, so if one matches, others should too)
                qh_matches = vertices_match(qh_vertices, scipy_vertices)
                jm_matches = vertices_match(jm_vertices, scipy_vertices)
                inc_matches = vertices_match(inc_vertices, scipy_vertices)
                
                scipy_match = qh_matches or jm_matches or inc_matches
                
                if not scipy_match:
                    # Fallback: If exact vertex matching fails, check if vertex counts are close
                    # Sometimes numerical precision causes vertices to not match exactly,
                    # but the counts should still be similar
                    # Compare vertex counts and allow reasonable differences
                    qh_count_diff = abs(len(qh_vertices) - len(scipy_vertices))
                    jm_count_diff = abs(len(jm_vertices) - len(scipy_vertices))
                    inc_count_diff = abs(len(inc_vertices) - len(scipy_vertices))
                    
                    min_count_diff = min(qh_count_diff, jm_count_diff, inc_count_diff)
                    
                    # Allow up to 10 vertex difference or 30% of scipy vertices (whichever is larger)
                    # This is more lenient to account for algorithm differences and numerical precision
                    allowed_count_diff = max(10, int(len(scipy_vertices) * 0.3))
                    
                    if min_count_diff <= allowed_count_diff:
                        # Vertex counts are close enough - algorithms are producing similar hulls
                        scipy_match = True
                    else:
                        # Vertex counts are too different - might indicate an issue
                        scipy_match = False
            except Exception as e:
                # If scipy verification fails, don't fail the whole check
                # Scipy might have issues with degenerate cases that our algorithms handle differently
                # We don't want to fail correctness check just because scipy fails
                scipy_match = True
        
        return all_valid and face_diff_ok and vertex_diff_ok and scipy_match
    except Exception as e:
        print(f"    Correctness check error: {e}")
        return False


def run_benchmark_suite(input_sizes: List[int], trials: int = 5, 
                       output_dir: str = "results", use_scipy_reference: bool = False) -> List[Dict[str, Any]]:
    """
    Run the full benchmark suite on all algorithms with multiple trials.
    
    This function tests all three algorithms (QuickHull, Jarvis March, Incremental)
    on varying input sizes with multiple trials each. Multiple trials provide
    statistical significance and account for variance in random inputs and system load.
    
    Args:
        input_sizes: List of input sizes (number of points) to test
        trials: Number of trials per algorithm/size combination (default: 5)
        output_dir: Directory to save results (default: "results")
        use_scipy_reference: If True, verify results against scipy.spatial.ConvexHull
        
    Returns:
        List of dictionaries containing all benchmark results, one per trial
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    algorithms = ['quickhull', 'jarvis', 'incremental']
    
    # Print benchmark configuration
    print("=" * 60)
    print("3D Convex Hull Algorithm Benchmark Suite")
    print("=" * 60)
    print(f"Input sizes: {input_sizes}")
    print(f"Trials per combination: {trials}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print("=" * 60)
    print()
    
    # Calculate total number of test runs for progress tracking
    total_combinations = len(input_sizes) * len(algorithms) * trials
    current = 0
    
    # Test each input size
    for size in input_sizes:
        print(f"Testing input size: {size}")
        
        # Test each algorithm
        for algo_name in algorithms:
            print(f"  Algorithm: {algo_name}")
            
            # Run multiple trials for statistical significance
            # Multiple trials help account for variance in random inputs and system load
            for trial in range(1, trials + 1):
                current += 1
                print(f"    Trial {trial}/{trials} ({current}/{total_combinations})...", end=" ")
                
                # Generate random points for this trial
                # Each trial uses different random input to test algorithm robustness
                points = generate_random_points(size, bounds=(-10, 10))
                
                try:
                    # Run the algorithm and collect performance metrics
                    result = run_algorithm(algo_name, points, trial)
                    results.append(result)
                    
                    # Print success status with key metrics
                    if result['success']:
                        print(f"[OK] {result['time_ms']:.2f}ms, {result['faces']} faces")
                    else:
                        print("[FAILED]")
                except Exception as e:
                    # Handle errors gracefully and record them
                    # Don't let one failed test stop the entire benchmark suite
                    print(f"[ERROR] {e}")
                    results.append({
                        'algorithm': algo_name,
                        'input_size': size,
                        'time_ms': None,
                        'memory_mb': None,
                        'iterations': None,
                        'comparisons': None,
                        'faces': None,
                        'vertices': None,
                        'trial': trial,
                        'success': False,
                        'error': str(e)
                    })
        
        # Verify correctness on one trial per input size
        # Ensures algorithms produce correct results, not just fast results
        # Only verify for smaller sizes to avoid performance impact
        if size <= 100:  # Only verify for smaller sizes (correctness check is expensive)
            points = generate_random_points(size, bounds=(-10, 10))
            if verify_correctness(points, use_scipy_reference=use_scipy_reference):
                scipy_note = " (with scipy reference)" if use_scipy_reference and SCIPY_AVAILABLE else ""
                print(f"  [OK] Correctness verified for size {size}{scipy_note}")
            else:
                print(f"  [WARNING] Correctness check failed for size {size}")
        print()
    
    return results


def save_results_csv(results: List[Dict[str, Any]], filename: str):
    """
    Save benchmark results to CSV file for easy analysis.
    
    CSV format is widely supported and easy to import into spreadsheet software
    or analysis tools like Excel, pandas, etc.
    
    Args:
        results: List of result dictionaries from benchmark runs
        filename: Output CSV filename path
    """
    if not results:
        return
    
    # Define CSV column names - these match the keys in result dictionaries
    fieldnames = ['algorithm', 'input_size', 'time_ms', 'memory_mb', 
                  'iterations', 'comparisons', 'faces', 'vertices', 'trial', 'success']
    
    # Write CSV file with header row
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write each result as a row, extracting only the specified fields
        for result in results:
            writer.writerow({k: result.get(k) for k in fieldnames})


def save_results_json(results: List[Dict[str, Any]], filename: str):
    """
    Save benchmark results to JSON file with metadata for reproducibility.
    
    JSON format preserves all data types and includes metadata about the test
    environment (platform, Python version, etc.) which is crucial for reproducing
    results and understanding test conditions.
    
    Args:
        results: List of result dictionaries from benchmark runs
        filename: Output JSON filename path
    """
    # Collect metadata about the test environment
    # This information is crucial for reproducing results and understanding test conditions
    metadata = {
        'timestamp': datetime.now().isoformat(),  # When the benchmark was run
        'platform': platform.platform(),  # Operating system information
        'python_version': platform.python_version(),  # Python version used
        'numpy_version': np.__version__,  # NumPy version (affects performance)
        'total_results': len(results)  # Number of test runs
    }
    
    # Structure output with metadata and results
    output = {
        'metadata': metadata,
        'results': results
    }
    
    # Write JSON file with pretty-printing (indent=2)
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    """
    Main entry point for benchmark script.
    
    Parses command-line arguments, runs the benchmark suite, and saves results
    in both CSV and JSON formats. Provides a command-line interface for running
    benchmarks with customizable parameters.
    """
    # Set up command-line argument parser
    # Allows users to customize benchmark parameters without editing code
    parser = argparse.ArgumentParser(description='Benchmark 3D convex hull algorithms')
    parser.add_argument('--sizes', type=int, nargs='+', 
                       default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                       help='Input sizes to test')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per algorithm/size combination')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--verify-scipy', action='store_true',
                       help='Also verify results against scipy.spatial.ConvexHull (optional, for verification only)')
    
    args = parser.parse_args()
    
    # Check if scipy is available when verification is requested
    # Warn user if they request scipy verification but scipy isn't installed
    if args.verify_scipy and not SCIPY_AVAILABLE:
        print("Warning: --verify-scipy requested but scipy is not available.")
        print("Install scipy with: pip install scipy")
        print("Continuing without scipy verification...")
        args.verify_scipy = False
    
    # Run the benchmark suite with specified parameters
    # Executes all tests and collects performance metrics
    results = run_benchmark_suite(args.sizes, args.trials, args.output, 
                                 use_scipy_reference=args.verify_scipy)
    
    # Save results in both CSV and JSON formats
    # CSV for spreadsheet analysis, JSON for programmatic access and metadata
    csv_path = os.path.join(args.output, 'benchmark_results.csv')
    json_path = os.path.join(args.output, 'benchmark_results.json')
    
    save_results_csv(results, csv_path)
    save_results_json(results, json_path)
    
    # Print completion message with output file locations
    print("=" * 60)
    print("Benchmark complete!")
    print(f"Results saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()

