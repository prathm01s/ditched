import numpy as np
import time
import os

# --- Configuration ---
N = 1_000_000
OUTPUT_DIR = "datasets"

# --- Helper Function ---

def save_points(data, filename, directory=OUTPUT_DIR):
    """Saves a 2D numpy array to a 'x y' text file compatible with C++."""
    
    filepath = os.path.join(directory, filename)
    print(f"    Saving {data.shape[0]:,} points to {filepath}...")
    
    # Save using SPACE delimiter and NO HEADER
    np.savetxt(
        filepath, 
        data, 
        delimiter=' ',  # <--- Changed to space
        fmt='%.8f'      # No header argument
    )

# --- Distribution Generators (1-9) ---

def gen_uniform_square(n):
    """1. Uniform Square: Basic sanity check."""
    print("Generating [1. Uniform Square]...")
    return np.random.rand(n, 2) * 1000

def gen_gaussian_blob(n):
    """2. Gaussian Blob: Average case, few hull points (low h)."""
    print("Generating [2. Gaussian Blob]...")
    mean = [500, 500]
    # Covariance for a rotated, elliptical blob
    cov = [[20000, 12000], 
           [12000, 15000]]
    return np.random.multivariate_normal(mean, cov, n)

def gen_multi_cluster(n):
    """3. Multi-Cluster: Tests bridging gaps between clusters."""
    print("Generating [3. Multi-Cluster]...")
    # Split N into three parts
    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2 # Remainder

    # Cluster 1: Top-left
    mean1 = [200, 800]
    cov1 = [[5000, 4000], [4000, 5000]]
    data1 = np.random.multivariate_normal(mean1, cov1, n1)
    
    # Cluster 2: Bottom-right
    mean2 = [800, 200]
    cov2 = [[3000, -1000], [-1000, 3000]]
    data2 = np.random.multivariate_normal(mean2, cov2, n2)
    
    # Cluster 3: Center (tighter)
    mean3 = [500, 500]
    cov3 = [[1000, 0], [0, 1000]]
    data3 = np.random.multivariate_normal(mean3, cov3, n3)
    
    return np.vstack((data1, data2, data3))

def gen_uniform_disk(n):
    """4. Uniform Disk: Best-case for Jarvis/Quickhull (low h)."""
    print("Generating [4. Uniform Disk]...")
    radius = 500
    center = [500, 500]
    
    # Generate random angles
    theta = 2 * np.pi * np.random.rand(n)
    
    # Use sqrt(random) for r to ensure uniform 2D distribution
    r = radius * np.sqrt(np.random.rand(n))
    
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    
    return np.column_stack((x, y))

def gen_circle_perimeter(n):
    """5. Circle Perimeter: Worst-case for Jarvis/Quickhull (h=N)."""
    print("Generating [5. Circle Perimeter]...")
    radius = 500
    center = [500, 500]
    
    theta = 2 * np.pi * np.random.rand(n)
    
    # Add a tiny bit of noise so no points are *truly* identical
    noise = np.random.normal(0, 0.01, n)
    
    x = center[0] + (radius + noise) * np.cos(theta)
    y = center[1] + (radius + noise) * np.sin(theta)
    
    return np.column_stack((x, y))

def gen_rectangle_perimeter(n):
    """6. Rectangle Perimeter: Strong collinearity test."""
    print("Generating [6. Rectangle Perimeter]...")
    min_x, max_x = 100, 900
    min_y, max_y = 100, 900
    
    # Split N across the four sides
    n_bottom = n // 4
    n_top = n // 4
    n_left = n // 4
    n_right = n - (n_bottom + n_top + n_left) # Ensure sum is N
    
    # Bottom edge: (rand_x, min_y)
    bottom = np.column_stack((np.random.uniform(min_x, max_x, n_bottom), np.full(n_bottom, min_y)))
    
    # Top edge: (rand_x, max_y)
    top = np.column_stack((np.random.uniform(min_x, max_x, n_top), np.full(n_top, max_y)))
    
    # Left edge: (min_x, rand_y)
    left = np.column_stack((np.full(n_left, min_x), np.random.uniform(min_y, max_y, n_left)))
    
    # Right edge: (max_x, rand_y)
    right = np.column_stack((np.full(n_right, max_x), np.random.uniform(min_y, max_y, n_right)))
    
    return np.vstack((bottom, top, left, right))

def gen_log_normal(n):
    """7. Log-Normal: Outlier robustness test."""
    print("Generating [7. Log-Normal]...")
    # Create a correlated 2D Gaussian
    mean = [0, 0]
    cov = [[1, 0.8],  # Add correlation to create a "comet" shape
           [0.8, 1]]
    data_normal = np.random.multivariate_normal(mean, cov, n)
    
    # Take the exponential to make it Log-Normal
    return np.exp(data_normal)

def gen_laplace(n):
    """8. Laplace: Non-convex cluster shape test."""
    print("Generating [8. Laplace]...")
    loc = 500  # Center
    scale = 100 # Spread
    
    x = np.random.laplace(loc, scale, n)
    y = np.random.laplace(loc, scale, n)
    
    return np.column_stack((x, y))

def gen_pareto(n):
    """9. Pareto: Extreme outlier and numerical stability test."""
    print("Generating [9. Pareto]...")
    a = 1.16  # Shape parameter (known as the "80/20" rule param)
    
    x = np.random.pareto(a, n)
    y = np.random.pareto(a, n)
    
    return np.column_stack((x, y))

# --- RENAMED gen_sorted_curve to gen_parabola ---
def gen_parabola(n):
    """10. Parabola (y=x^2). Best-case (post-sort) for Monotone Chain."""
    print("Generating [10. Parabola (Sorted Curve)]...")
    # Generate points sorted by x
    x = np.linspace(-500, 500, n)
    # Add a small amount of random noise to y
    y = (x/100)**2 + np.random.normal(0, 10, n)
    
    return np.column_stack((x, y))

def gen_fan(n):
    """11. Fan. Best-case (post-sort) for Graham Scan."""
    print("Generating [11. Fan (Sorted by Angle)]...")
    origin = [0, 0]
    
    # Generate points with angles already sorted
    angles = np.linspace(0, np.pi, n) # A 180-degree fan
    
    # Generate random distances
    radii = np.random.uniform(100, 1000, n)
    
    x = origin[0] + radii * np.cos(angles)
    y = origin[1] + radii * np.sin(angles)
    
    return np.column_stack((x, y))

def gen_spiral(n):
    """12. Archimedean Spiral. Worst-case (stack pops) for Graham Scan."""
    print("Generating [12. Spiral]...")
    # 'a' controls the distance between spiral arms
    a = 1
    # Generate angles that wind around multiple times
    theta = np.linspace(0, 10 * np.pi, n)
    
    # Add noise to theta to make points distinct
    theta_noisy = theta + np.random.normal(0, 0.05, n)
    
    # Archimedean spiral: r = a * theta
    r = a * theta_noisy
    
    x = r * np.cos(theta_noisy)
    y = r * np.sin(theta_noisy)
    
    return np.column_stack((x, y))

def gen_sawtooth(n):
    """13. Sawtooth. Worst-case (stack pops) for Monotone Chain."""
    print("Generating [13. Sawtooth]...")
    # Generate points sorted by x
    x = np.linspace(0, 1000, n)
    
    # Create a sawtooth wave for y
    amplitude = 100
    period = 50
    y = amplitude * (x / period - np.floor(x / period + 0.5))
    
    # Add noise to make it a 2D cloud
    y_noisy = y + np.random.normal(0, 5, n)
    
    return np.column_stack((x, y_noisy))

# --- RESEARCH-GRADE DISTRIBUTIONS (14-16) ---

def gen_grid(n):
    """14. Integer Grid: Extreme degeneracy (collinearity + duplicates)."""
    print("Generating [14. Integer Grid]...")
    # For N=1M, a 1000x1000 grid is a good test
    grid_size = 1000
    # Sample N points *with replacement* from a (grid_size+1) x (grid_size+1) grid
    x = np.random.randint(0, grid_size + 1, n)
    y = np.random.randint(0, grid_size + 1, n)
    # Save as float for consistency with other datasets
    return np.column_stack((x, y)).astype(float)

# --- RENAMED gen_almost_collinear to gen_needle ---
def gen_needle(n):
    """15. Needle (Almost Collinear): Tests floating-point precision."""
    print("Generating [15. Needle (Almost Collinear)]...")
    # X values are spread out
    x = np.random.uniform(0, 1000, n)
    # Y values are in a tiny, tiny range around 0
    y = np.random.normal(loc=0.0, scale=1e-9, size=n)
    return np.column_stack((x, y))

def gen_mandelbrot(n, max_iter=80, grid_res=1500):
    """16. Fractal Boundary: Complex structure, 'h' near 'N'."""
    print(f"Generating [16. Fractal (Mandelbrot)]... (This may take a moment)")
    
    # 1. Create a grid of complex numbers
    # We create a 1500x1500 grid, which is 2.25M points.
    x_range = np.linspace(-2.0, 1.0, grid_res)
    y_range = np.linspace(-1.5, 1.5, grid_res)
    xx, yy = np.meshgrid(x_range, y_range)
    c = xx + 1j * yy

    # 2. Initialize z and iteration counters
    z = np.zeros_like(c)
    iterations = np.zeros(c.shape, dtype=int)

    # 3. Run the iteration (vectorized)
    for i in range(max_iter):
        # Find points that have not escaped
        not_escaped = np.abs(z) <= 2
        # Update z for those points
        z[not_escaped] = z[not_escaped]**2 + c[not_escaped]
        # Update iteration count for those points
        iterations[not_escaped] += 1
    
    # 4. Find the boundary points
    # The boundary is where the iteration count is high, but not max
    # (i.e., it escaped, but very slowly). This creates the "glowing" edge.
    boundary_mask = (iterations > 10) & (iterations < max_iter)
    
    boundary_points_x = xx[boundary_mask]
    boundary_points_y = yy[boundary_mask]
    
    all_boundary_points = np.column_stack((boundary_points_x, boundary_points_y))
    
    # 5. Sample N points from this boundary set
    num_found = all_boundary_points.shape[0]
    
    if num_found == 0:
        print("    WARNING: Mandelbrot generation found 0 points. Returning empty.")
        return np.empty((0, 2))
    
    print(f"    Found {num_found:,} boundary points, sampling {n:,}...")
    
    if num_found < n:
        # Repeat points if we don't have enough
        indices = np.random.choice(num_found, n, replace=True)
    else:
        # Sample N points if we have more than enough
        indices = np.random.choice(num_found, n, replace=False)
    
    return all_boundary_points[indices]

# --- NEWLY ADDED DISTRIBUTIONS ---

def gen_poisson_process(n):
    """17. Poisson Process: Models complete spatial randomness."""
    print("Generating [17. Poisson Process]...")
    # For a large N, a Poisson process with mean N in a fixed area
    # is statistically indistinguishable from N points sampled uniformly.
    # We use this standard model of "complete spatial randomness" (CSR).
    return np.random.rand(n, 2) * 1000

def gen_onion(n, layers=10):
    """18. The Onion: Tests layer-peeling and stacked concavities."""
    print(f"Generating [18. The Onion]... ({layers} layers)")
    if layers == 0:
        return np.empty((0, 2))
        
    n_per_layer = n // layers
    all_layers = []
    
    for i in range(layers):
        # Create layers from outside-in
        radius = 500 * (1 - i / layers)
        
        # Make layers slightly offset to avoid perfect alignment
        center_x = np.random.uniform(-5, 5)
        center_y = np.random.uniform(-5, 5)
        
        # Calculate n for this layer, giving remainder to the last one
        current_n = n_per_layer if i < layers - 1 else n - (n_per_layer * (layers - 1))
        
        theta = 2 * np.pi * np.random.rand(current_n)
        
        # Add a bit of noise to the perimeter
        noise = np.random.normal(0, 0.5, current_n)
        
        x = center_x + (radius + noise) * np.cos(theta)
        y = center_y + (radius + noise) * np.sin(theta)
        
        all_layers.append(np.column_stack((x, y)))
    
    return np.vstack(all_layers)


# --- Main Execution ---

def main():
    """Generates all datasets and saves them to the OUTPUT_DIR."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"--- Starting Dataset Generation (N = {N:,}) ---")
    start_total = time.time()
    
    generators = {
        "case_uniform_square.txt": gen_uniform_square,
        "case_gaussian_blob.txt": gen_gaussian_blob,
        "case_multi_cluster.txt": gen_multi_cluster,
        "case_uniform_disk.txt": gen_uniform_disk,
        "case_circle_perimeter.txt": gen_circle_perimeter,
        "case_rectangle_perimeter.txt": gen_rectangle_perimeter,
        "case_log_normal.txt": gen_log_normal,
        "case_laplace.txt": gen_laplace,
        "case_pareto.txt": gen_pareto,
        "case_parabola.txt": gen_parabola, # Renamed
        "case_fan.txt": gen_fan,
        "case_spiral.txt": gen_spiral,
        "case_sawtooth.txt": gen_sawtooth,
        "case_grid.txt": gen_grid,
        "case_needle.txt": gen_needle, # Renamed
        "case_mandelbrot.txt": gen_mandelbrot,
        "case_poisson_process.txt": gen_poisson_process, # New
        "case_onion.txt": gen_onion, # New
    }
    
    for filename, func in generators.items():
        start_gen = time.time()
        # Generate the data
        data = func(N)
        
        if data.shape[0] > 0:
            # Shuffle the data before saving
            # This ensures that pre-sorted datasets (like Parabola, Fan)
            # are still a valid test for the algorithm's sorting step.
            np.random.shuffle(data)
            
            # Save the data
            save_points(data, filename, directory=OUTPUT_DIR)
        else:
            print(f"    Skipping save for {filename} (0 points generated).")

        end_gen = time.time()
        print(f"    Finished in {end_gen - start_gen:.2f} seconds.\n")

    end_total = time.time()
    print(f"--- All {len(generators)} datasets generated in {end_total - start_total:.2f} seconds ---")

if __name__ == "__main__":
    main()