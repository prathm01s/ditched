import numpy as np
import matplotlib
# Set 'Agg' backend to tell matplotlib we are not showing plots, just saving.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_data(filename, sample_size=10000):
    """
    Loads and plots a 2D dataset, using sampling for large files.
    """
    # Quietly return if file doesn't exist to prevent crashing parallel workers
    if not os.path.exists(filename):
        return

    try:
        # Load the data, skipping the 'x,y' header
        data = np.loadtxt(filename, skiprows=1, delimiter=',')
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return

    # Handle empty files or single points
    if data.ndim == 1:
        data = data.reshape(-1, 2)
        
    n_total = data.shape[0]
    
    if n_total == 0:
        return

    # --- Sampling ---
    if n_total > sample_size:
        # Get random indices to sample
        indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
        sampled_data = data[indices]
        title_suffix = f"(Sampled {sample_size:,} of {n_total:,})"
    else:
        sampled_data = data
        title_suffix = f"({n_total:,} points)"

    x = sampled_data[:, 0]
    y = sampled_data[:, 1]

    # --- Plotting ---
    plt.figure(figsize=(10, 10))
    
    # Use a small point size (s=1) and transparency (alpha)
    # for a good "density" view
    plt.scatter(x, y, s=1, alpha=0.3)
    
    # Set axis to 'equal' to see true geometric shapes
    plt.axis('equal')
    
    base_name = os.path.basename(filename)
    plt.title(f"Dataset: {base_name}\n{title_suffix}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # --- Save the plot ---
    output_filename = os.path.splitext(filename)[0] + ".png"
    
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        # print(f"Plot saved: {output_filename}") # Commented out to reduce console spam
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        # CRITICAL: Close the figure to free memory!
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 2D point datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "filename", 
        type=str, 
        help="The path to the .txt dataset file."
    )
    
    parser.add_argument(
        "--sample", 
        type=int, 
        default=20000, 
        help="The number of points to randomly sample for plotting. Default: 20,000"
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File not found at {args.filename}")
    else:
        plot_data(args.filename, sample_size=args.sample)