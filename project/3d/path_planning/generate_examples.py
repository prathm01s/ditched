"""
Generate example scenarios for visualization.

This script generates multiple example JSON files for the visualization.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from path_planning.path_planner import PathPlanner
from path_planning.visualizer.data_export import export_path_data

examples = {
    "simple": {
        "name": "Simple Path",
        "num_obstacles": 3,
        "obstacle_types": ["cube"],
        "seed": 1
    },
    "maze": {
        "name": "Complex Maze",
        "num_obstacles": 10,  # Reduced from 15 for faster generation
        "obstacle_types": ["cube", "sphere"],
        "seed": 42
    },
    "narrow": {
        "name": "Narrow Passage",
        "num_obstacles": 8,  # Reduced from 12
        "obstacle_types": ["cube"],
        "seed": 100
    },
    "high": {
        "name": "High Obstacles",
        "num_obstacles": 6,  # Reduced from 8
        "obstacle_types": ["cube"],
        "min_cube_size": 3.0,
        "max_cube_size": 6.0,
        "seed": 200
    },
    "spherical": {
        "name": "Spherical Challenge",
        "num_obstacles": 8,  # Reduced from 10
        "obstacle_types": ["sphere"],
        "min_sphere_radius": 1.5,
        "max_sphere_radius": 3.5,
        "seed": 300
    },
    "mixed": {
        "name": "Mixed Environment",
        "num_obstacles": 8,  # Reduced from 12
        "obstacle_types": ["cube", "sphere"],
        "seed": 500
    }
}

def generate_all_examples():
    """Generate all example scenarios."""
    output_dir = "path_planning/visualizer/examples"
    os.makedirs(output_dir, exist_ok=True)
    
    for key, config in examples.items():
        print(f"\nGenerating {config['name']}...")
        
        planner = PathPlanner(
            workspace_size=(20, 20, 20),
            start_pos=(0, 0, 0),
            goal_pos=(19, 19, 19)
        )
        
        # Generate obstacles
        obstacles = planner.generate_obstacles(
            num_obstacles=config["num_obstacles"],
            obstacle_types=config["obstacle_types"],
            min_cube_size=config.get("min_cube_size", 2.0),
            max_cube_size=config.get("max_cube_size", 5.0),
            min_sphere_radius=config.get("min_sphere_radius", 1.0),
            max_sphere_radius=config.get("max_sphere_radius", 3.0),
            seed=config["seed"]
        )
        
        # Plan path
        path = planner.plan_path()
        
        if path:
            output_file = os.path.join(output_dir, f"{key}.json")
            export_path_data(planner, output_file)
            print(f"  [OK] Path found: {len(path)} nodes, {planner.get_path_length():.2f} units")
            print(f"  [OK] Saved to: {output_file}")
        else:
            print(f"  [FAIL] No path found for {config['name']}")
    
    print(f"\n[OK] All examples generated in {output_dir}/")

if __name__ == "__main__":
    generate_all_examples()

