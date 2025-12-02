import numpy as np
import time
import os
import argparse

# --- Configuration ---
OUTPUT_DIR = "datasets"

# --- Helper Function ---

def save_points(data, filename, directory=OUTPUT_DIR):
    """
    Save a 2D numpy array of points to a CSV file with x,y format.
    
    Args:
        data (numpy.ndarray): Nx2 array of 2D points to save.
        filename (str): Name of the output file.
        directory (str): Directory to save the file. Defaults to OUTPUT_DIR.
    
    Returns:
        None. Saves file to <directory>/<filename>.
    """
    
    filepath = os.path.join(directory, filename)
    print(f"    Saving {data.shape[0]:,} points to {filepath}...")
    
    # Save with 8 decimal places of precision
    np.savetxt(
        filepath, 
        data, 
        delimiter=',',
        fmt='%.8f',
        header='x,y',
        comments='' # Removes the '#' from the header line
    )

# --- Distribution Generators ---

def gen_uniform_square(n):
    """
    Generate uniformly distributed points in a square.
    Basic sanity check distribution with moderate hull size.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points uniformly distributed in [0, 1000] x [0, 1000].
    """
    print("Generating [1. Uniform Square]...")
    return np.random.rand(n, 2) * 1000

def gen_gaussian_blob(n):
    """
    Generate points from a Gaussian (normal) distribution centered at (500, 500).
    Average case distribution with few hull points (low h) - best case for output-sensitive algorithms.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points from multivariate normal distribution.
    """
    print("Generating [2. Gaussian Blob]...")
    mean = [500, 500]
    cov = [[20000, 12000], 
           [12000, 15000]]
    return np.random.multivariate_normal(mean, cov, n)

def gen_clusters_specific(n, num_clusters):
    """
    Generate N points divided among multiple Gaussian clusters.
    Tests algorithm performance on multi-modal distributions with multiple convex regions.
    
    Args:
        n (int): Total number of points to generate.
        num_clusters (int): Number of separate Gaussian clusters.
    
    Returns:
        numpy.ndarray: Nx2 array of points from num_clusters different Gaussian blobs.
    """
    print(f"Generating [3. Multi-Cluster] with {num_clusters} clusters...")
    
    points_per_cluster = n // num_clusters
    data_list = []
    
    for _ in range(num_clusters):
        # Random center within the field
        center = np.random.randint(0, 1000, 2)
        # Random spread for this cluster
        spread = np.random.randint(10, 100)
        cov = [[spread*10, 0], [0, spread*10]]
        
        cluster_data = np.random.multivariate_normal(center, cov, points_per_cluster)
        data_list.append(cluster_data)
    
    # Handle remainder points
    remainder = n - (points_per_cluster * num_clusters)
    if remainder > 0:
        center = np.random.randint(0, 1000, 2)
        cov = [[100, 0], [0, 100]]
        data_list.append(np.random.multivariate_normal(center, cov, remainder))
        
    return np.vstack(data_list)

def gen_uniform_disk(n):
    """
    Generate uniformly distributed points within a circular disk.
    Best-case for Jarvis March/QuickHull - circular convex hull with low h.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points uniformly distributed inside a circle.
    """
    print("Generating [4. Uniform Disk]...")
    radius = 500
    center = [500, 500]
    theta = 2 * np.pi * np.random.rand(n)
    r = radius * np.sqrt(np.random.rand(n))
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))

def gen_circle_perimeter(n):
    """
    Generate points on the perimeter of a circle with small noise.
    Worst-case for Jarvis March/QuickHull - all points on hull (h=N).
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points on a circle boundary with slight noise.
    """
    print("Generating [5. Circle Perimeter]...")
    radius = 500
    center = [500, 500]
    theta = 2 * np.pi * np.random.rand(n)
    noise = np.random.normal(0, 0.01, n)
    x = center[0] + (radius + noise) * np.cos(theta)
    y = center[1] + (radius + noise) * np.sin(theta)
    return np.column_stack((x, y))

def gen_rectangle_perimeter(n):
    """
    Generate points on the perimeter of a rectangle.
    Strong collinearity test - many collinear points on each edge.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points distributed on rectangle edges.
    """
    print("Generating [6. Rectangle Perimeter]...")
    min_x, max_x = 100, 900
    min_y, max_y = 100, 900
    n_side = n // 4
    n_rem = n - (n_side * 3)
    
    bottom = np.column_stack((np.random.uniform(min_x, max_x, n_side), np.full(n_side, min_y)))
    top = np.column_stack((np.random.uniform(min_x, max_x, n_side), np.full(n_side, max_y)))
    left = np.column_stack((np.full(n_side, min_x), np.random.uniform(min_y, max_y, n_side)))
    right = np.column_stack((np.full(n_rem, max_x), np.random.uniform(min_y, max_y, n_rem)))
    
    return np.vstack((bottom, top, left, right))

def gen_log_normal(n):
    """
    Generate points from a log-normal distribution.
    Tests outlier robustness with heavy-tailed distribution.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points from log-normal distribution.
    """
    print("Generating [7. Log-Normal]...")
    mean = [2, 2] # Shifted mean to spread it out
    cov = [[1, 0.5], [0.5, 1]]
    data_normal = np.random.multivariate_normal(mean, cov, n)
    # Scale output to spread points more visibly
    return np.exp(data_normal) * 10

def gen_laplace(n):
    """
    Generate points from a Laplace (double exponential) distribution.
    Tests non-convex cluster shapes.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points from Laplace distribution.
    """
    print("Generating [8. Laplace]...")
    loc = 500
    scale = 100
    x = np.random.laplace(loc, scale, n)
    y = np.random.laplace(loc, scale, n)
    return np.column_stack((x, y))

def gen_pareto(n):
    """
    Generate points from a Pareto distribution.
    Extreme outlier test with very heavy tails.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points from Pareto distribution scaled by 1000.
    """
    print("Generating [9. Pareto]...")
    a = 1.16 
    # Scale by 1000 so the "compact" part is visible and tails are huge
    x = np.random.pareto(a, n) * 1000 
    y = np.random.pareto(a, n) * 1000
    return np.column_stack((x, y))

def gen_parabola(n):
    """
    Generate points along a parabolic curve y = x².
    Tests algorithms on curved, collinear-like patterns.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points along a parabola with noise.
    """
    print("Generating [10. Parabola]...")
    x = np.linspace(-500, 500, n)
    y = (x/100)**2 * 100 + np.random.normal(0, 10, n)
    return np.column_stack((x, y))

def gen_fan(n):
    """
    Generate points in a fan (half-circle) pattern radiating from origin.
    Tests angular distribution patterns.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points in a fan shape.
    """
    print("Generating [11. Fan]...")
    origin = [0, 0]
    angles = np.linspace(0, np.pi, n)
    radii = np.random.uniform(100, 1000, n)
    x = origin[0] + radii * np.cos(angles)
    y = origin[1] + radii * np.sin(angles)
    return np.column_stack((x, y))

def gen_spiral(n):
    """
    Generate points along an Archimedean spiral.
    Tests algorithms on spiral/curved hull boundaries.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points forming a spiral pattern.
    """
    print("Generating [12. Spiral]...")
    a = 1
    theta = np.linspace(0, 20 * np.pi, n)
    theta_noisy = theta + np.random.normal(0, 0.05, n)
    r = a * theta_noisy
    x = r * np.cos(theta_noisy)
    y = r * np.sin(theta_noisy)
    return np.column_stack((x, y))

def gen_sawtooth(n):
    """
    Generate points following a sawtooth (zigzag) pattern.
    Tests edge detection and sharp angle handling.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points in a sawtooth pattern with noise.
    """
    print("Generating [13. Sawtooth]...")
    x = np.linspace(0, 1000, n)
    amplitude = 100
    period = 50
    y = amplitude * (x / period - np.floor(x / period + 0.5))
    y_noisy = y + np.random.normal(0, 5, n)
    return np.column_stack((x, y_noisy))

def gen_grid(n):
    """
    Generate points on an integer grid.
    Degeneracy testing with many potential duplicate or near-duplicate points.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points on integer lattice positions.
    """
    print("Generating [14. Integer Grid]...")
    grid_size = int(np.sqrt(n)) # Dynamic grid size based on N
    x = np.random.randint(0, grid_size, n)
    y = np.random.randint(0, grid_size, n)
    return np.column_stack((x, y)).astype(float)

def gen_needle(n):
    """
    Generate points along a thin horizontal line (needle).
    Extreme collinearity test with points in an almost 1D distribution.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points forming a thin horizontal line.
    """
    print("Generating [15. Needle]...")
    x = np.random.uniform(0, 1000, n)
    y = np.random.normal(loc=0.0, scale=1e-9, size=n)
    return np.column_stack((x, y))

def gen_mandelbrot(n, max_iter=80, grid_res=1500):
    """
    Generate points sampled from the boundary of the Mandelbrot set.
    Fractal boundary test with complex, intricate hull structure.
    
    Args:
        n (int): Number of points to generate.
        max_iter (int): Maximum iterations for Mandelbrot computation. Defaults to 80.
        grid_res (int): Grid resolution for sampling. Defaults to 1500.
    
    Returns:
        numpy.ndarray: Nx2 array of points from Mandelbrot set boundary.
    """
    print(f"Generating [16. Fractal (Mandelbrot)]...")
    x_range = np.linspace(-2.0, 1.0, grid_res)
    y_range = np.linspace(-1.5, 1.5, grid_res)
    xx, yy = np.meshgrid(x_range, y_range)
    c = xx + 1j * yy
    z = np.zeros_like(c)
    iterations = np.zeros(c.shape, dtype=int)

    for i in range(max_iter):
        not_escaped = np.abs(z) <= 2
        z[not_escaped] = z[not_escaped]**2 + c[not_escaped]
        iterations[not_escaped] += 1
    
    boundary_mask = (iterations > 10) & (iterations < max_iter)
    boundary_points_x = xx[boundary_mask]
    boundary_points_y = yy[boundary_mask]
    all_boundary_points = np.column_stack((boundary_points_x, boundary_points_y))
    
    num_found = all_boundary_points.shape[0]
    if num_found == 0: return np.empty((0, 2))
    
    if num_found < n:
        indices = np.random.choice(num_found, n, replace=True)
    else:
        indices = np.random.choice(num_found, n, replace=False)
    return all_boundary_points[indices]

def gen_poisson_process(n):
    """
    Generate points from a spatial Poisson process (uniformly random).
    Tests truly random spatial distribution.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of uniformly random points in [0, 1000] x [0, 1000].
    """
    print("Generating [17. Poisson Process]...")
    return np.random.rand(n, 2) * 1000

def gen_onion(n):
    """
    Generate points on multiple concentric circles (onion layers).
    Tests algorithms on nested boundary structures.
    
    Args:
        n (int): Number of points to generate.
    
    Returns:
        numpy.ndarray: Nx2 array of points distributed across 50 concentric circles.
    """
    # Scale layers based on N, but keep it reasonable (e.g., 50 layers for 1M points)
    layers = 50 
    print(f"Generating [18. The Onion] with {layers} layers...")
    
    n_per_layer = n // layers
    all_layers = []
    
    for i in range(layers):
        radius = 500 * (1 - i / layers)
        # Small jitter to centers
        center_x = np.random.uniform(-2, 2)
        center_y = np.random.uniform(-2, 2)
        
        current_n = n_per_layer if i < layers - 1 else n - (n_per_layer * (layers - 1))
        theta = 2 * np.pi * np.random.rand(current_n)
        noise = np.random.normal(0, 0.2, current_n)
        
        x = center_x + (radius + noise) * np.cos(theta)
        y = center_y + (radius + noise) * np.sin(theta)
        all_layers.append(np.column_stack((x, y)))
    
    return np.vstack(all_layers)

# --- Main Execution ---

def main():
    """
    Main execution function for dataset generation.
    Parses command-line arguments and generates all 18 distributions.
    
    Args:
        Command-line argument N (int, optional): Number of points per dataset. Default: 1,000,000.
    
    Returns:
        None. Saves all generated datasets to the datasets/ directory.
    """
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Generate 2D point datasets.")
    parser.add_argument("N", type=int, nargs="?", default=1_000_000, 
                        help="Number of points to generate (default: 1,000,000)")
    args = parser.parse_args()
    
    N_points = args.N

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"--- Starting Dataset Generation (N = {N_points:,}) ---")
    start_total = time.time()
    
    # Standard Generators
    generators = {
        "1_uniform_square.txt": gen_uniform_square,
        "2_gaussian_blob.txt": gen_gaussian_blob,
        "4_uniform_disk.txt": gen_uniform_disk,
        "5_circle_perimeter.txt": gen_circle_perimeter,
        "6_rectangle_perimeter.txt": gen_rectangle_perimeter,
        "7_log_normal.txt": gen_log_normal,
        "8_laplace.txt": gen_laplace,
        "9_pareto.txt": gen_pareto,
        "10_parabola.txt": gen_parabola,
        "11_fan.txt": gen_fan,
        "12_spiral.txt": gen_spiral,
        "13_sawtooth.txt": gen_sawtooth,
        "14_grid.txt": gen_grid,
        "15_needle.txt": gen_needle,
        "16_mandelbrot.txt": gen_mandelbrot,
        "17_poisson_process.txt": gen_poisson_process,
        "18_onion.txt": gen_onion,
    }
    
    # 1. Run Standard Generators
    for filename, func in generators.items():
        start_gen = time.time()
        data = func(N_points)
        if data.shape[0] > 0:
            np.random.shuffle(data)
            save_points(data, filename)
        print(f"    Finished in {time.time() - start_gen:.2f} s\n")

    # 2. Generate Multi-Cluster with 200 clusters (5 instances for statistical variance)
    num_instances = 5
    cluster_count = 200
    for instance in range(1, num_instances + 1):
        start_gen = time.time()
        filename = f"3_multi_cluster_200_instance_{instance}.txt"
        data = gen_clusters_specific(N_points, cluster_count)
        np.random.shuffle(data)
        save_points(data, filename)
        print(f"    Finished in {time.time() - start_gen:.2f} s\n")

    print(f"--- All datasets generated in {time.time() - start_total:.2f} seconds ---")

if __name__ == "__main__":
    main()