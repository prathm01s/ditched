import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import os # <-- NEW IMPORT

# Define the directory name
DATASET_DIR = "datasets"

# --- Utility Function (Updated for folder) ---
def save_points(points, filename):
    """Saves an (n, 2) numpy array to a text file in the DATASET_DIR."""
    
    # Construct the full path
    full_path = os.path.join(DATASET_DIR, filename)

    # Ensure the file is not written if points is empty
    if points.size == 0:
         print(f"Generated: {full_path} (0 points) - EMPTY")
         with open(full_path, 'w') as f:
             f.write('')
         return

    # Use space delimiter
    np.savetxt(full_path, points, fmt='%.8f', delimiter=' ')
    print(f"Generated: {full_path} ({len(points)} points)")

# --- Data Generation Functions (Unchanged Logic, now use modified save_points) ---

def gen_case_sorted(n=100000):
    """Case: Points are perfectly sorted by X, then Y (Best case for initial sort)."""
    x = np.linspace(0, 100, n)
    y = np.sin(x / 5) + np.random.uniform(-0.1, 0.1, n)
    points = np.stack([x, y], axis=1)
    save_points(points, "case_perfectly_sorted.txt")

def gen_case_explicit_duplicates(n=100000):
    """Case: Only a few unique, exactly identical points, repeated many times."""
    base_points = np.array([
        [10.0, 10.0], [5.0, 2.0], [1.0, 7.0], [8.0, 8.0], [3.0, 3.0]
    ])
    indices = np.random.randint(0, 5, size=n)
    points = base_points[indices]
    save_points(points, "case_explicit_duplicates.txt")

def gen_case_triangle_edge(n=100000):
    """Case: Points scattered only on the edges of a simple triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([10.0, 0.0])
    C = np.array([5.0, 10.0])
    t = np.random.rand(n)
    n_per_side = n // 3
    p1 = A + (B - A) * t[:n_per_side].reshape(-1, 1)
    p2 = B + (C - B) * t[n_per_side:2*n_per_side].reshape(-1, 1)
    p3 = C + (A - C) * t[2*n_per_side:].reshape(-1, 1)
    points = np.concatenate([p1, p2, p3])
    save_points(points, "case_triangle_edge.txt")

def gen_case_uniform_random(n=100000):
    """Case: Large, standard random set (uniform distribution)."""
    points = np.random.uniform(-100, 100, (n, 2))
    save_points(points, "case_uniform_random.txt")

def gen_case_collinear_vertical(n=100000):
    """Case: All points on a perfect vertical line."""
    x = np.ones(n)
    y = np.linspace(0, 10, n)
    np.random.shuffle(y)
    points = np.stack([x, y], axis=1)
    save_points(points, "case_collinear_vertical.txt")

def gen_case_collinear_horizontal(n=100000):
    """Case: All points on a perfect horizontal line."""
    x = np.linspace(0, 10, n)
    y = np.ones(n)
    points = np.stack([x, y], axis=1)
    save_points(points, "case_collinear_horizontal.txt")

def gen_case_gaussian_blob(n=100000):
    """Case: Standard "best case" cloud of points."""
    points, _ = make_blobs(n_samples=n, centers=1, cluster_std=50.0, random_state=42)
    save_points(points, "case_gaussian_blob.txt")

def gen_case_circle(n=100000):
    """Case: Points on a circle. (Worst case for h)"""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = 100 * np.column_stack([np.cos(theta), np.sin(theta)])
    points += np.random.randn(n, 2) * 0.001 
    save_points(points, "case_circle_worst.txt")

def gen_case_u_shape(n=200):
    """Case: "U" shape (tests lower hull logic)"""
    points, _ = make_moons(n_samples=n, noise=0.05, random_state=6)
    save_points(points, "case_u_shape.txt")

def gen_case_rectangle(n=100000):
    """Case: Points in a rectangle. (Tests 4 corners, interior points)"""
    corners = np.array([
        [0, 0], [10, 0], [10, 5], [0, 5]
    ])
    interior = np.random.rand(n - 4, 2)
    interior[:, 0] *= 9.99    
    interior[:, 1] *= 4.99    
    interior += 0.005
    points = np.concatenate([corners, interior])
    save_points(points, "case_rectangle_large.txt")
    
def gen_case_small_duplicates(n=100):
    """Case: Only a few unique points, repeated many times (for small-scale testing)."""
    base_points = np.array([
        [0, 0], [5, 2], [1, 7], [8, 8]
    ])
    indices = np.random.randint(0, 4, size=n)
    points = base_points[indices]
    points = points + np.random.randn(n, 2) * 1e-5
    save_points(points, "case_duplicates_small.txt")


if __name__ == "__main__":
    
    # 0. Create the datasets folder if it doesn't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created directory: {DATASET_DIR}/")

    print("\n--- Generating Test Datasets for Convex Hull (N=100000) ---")
    
    # 1. Collinearity Tests
    gen_case_collinear_vertical(100000)
    gen_case_collinear_horizontal(100000)
    
    # 2. Performance and Distribution Tests (N=100000)
    gen_case_uniform_random(100000)
    gen_case_gaussian_blob(100000)
    gen_case_circle(100000)
    gen_case_rectangle(100000)
    
    # 3. Sorting and Duplicates Tests (N=100000)
    gen_case_sorted(100000)
    gen_case_explicit_duplicates(100000)
    
    # 4. Geometric Edge Cases (N=100000 and N=200)
    gen_case_triangle_edge(100000)
    gen_case_u_shape(200)
    
    # 5. Trivial/Small Cases
    save_points(np.array([[5.0, 5.0]]), "case_single_point.txt")
    save_points(np.array([[5.0, 5.0], [5.0, 6.0]]), "case_two_points.txt")
    save_points(np.array([]).reshape(0, 2), "case_empty_set.txt")
    gen_case_small_duplicates(100)
    
    print("\n--- All 13 datasets generated and saved in the 'datasets' folder. ---")