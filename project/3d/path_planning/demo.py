"""
Demo Script for 3D Robot Path Planning

Command-line interface for running path planning demos with various options.
"""

import argparse
import sys
import os
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from path_planning.path_planner import PathPlanner
from path_planning.visualizer.data_export import export_path_data


def main():
    """Main entry point for demo script."""
    parser = argparse.ArgumentParser(
        description='3D Robot Path Planning & Obstacle Avoidance Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic demo with 8 obstacles
  python demo.py
  
  # Demo with 10 obstacles, only cubes
  python demo.py --num-obstacles 10 --obstacle-types cube
  
  # Demo with custom workspace size
  python demo.py --workspace-size 15 15 15
  
  # Demo with visualization export
  python demo.py --export path_data.json --visualize
        """
    )
    
    parser.add_argument('--num-obstacles', type=int, default=8,
                       help='Number of obstacles to generate (default: 8)')
    parser.add_argument('--obstacle-types', nargs='+', 
                       choices=['cube', 'sphere'],
                       default=['cube', 'sphere'],
                       help='Types of obstacles to generate (default: cube sphere)')
    parser.add_argument('--workspace-size', type=int, nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=[20, 20, 20],
                       help='Workspace size in x, y, z (default: 20 20 20)')
    parser.add_argument('--start', type=int, nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=[0, 0, 0],
                       help='Start position (default: 0 0 0)')
    parser.add_argument('--goal', type=int, nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Goal position (default: workspace_size - 1)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for obstacle generation')
    parser.add_argument('--export', type=str, default=None,
                       help='Export path data to JSON file for visualization')
    parser.add_argument('--visualize', action='store_true',
                       help='Start visualization server after export')
    parser.add_argument('--no-diagonals', action='store_true',
                       help='Use 6-connected neighbors instead of 26-connected')
    parser.add_argument('--min-cube-size', type=float, default=2.0,
                       help='Minimum cube size (default: 2.0)')
    parser.add_argument('--max-cube-size', type=float, default=5.0,
                       help='Maximum cube size (default: 5.0)')
    parser.add_argument('--min-sphere-radius', type=float, default=1.0,
                       help='Minimum sphere radius (default: 1.0)')
    parser.add_argument('--max-sphere-radius', type=float, default=3.0,
                       help='Maximum sphere radius (default: 3.0)')
    
    args = parser.parse_args()
    
    # Set default goal if not provided
    if args.goal is None:
        args.goal = [args.workspace_size[0] - 1, 
                    args.workspace_size[1] - 1, 
                    args.workspace_size[2] - 1]
    
    print("=" * 60)
    print("3D Robot Path Planning & Obstacle Avoidance Demo")
    print("=" * 60)
    print(f"Workspace size: {args.workspace_size[0]}x{args.workspace_size[1]}x{args.workspace_size[2]}")
    print(f"Start position: {args.start}")
    print(f"Goal position: {args.goal}")
    print(f"Number of obstacles: {args.num_obstacles}")
    print(f"Obstacle types: {', '.join(args.obstacle_types)}")
    print(f"Neighbors: {'6-connected' if args.no_diagonals else '26-connected'}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    # Create path planner
    planner = PathPlanner(
        workspace_size=tuple(args.workspace_size),
        start_pos=tuple(args.start),
        goal_pos=tuple(args.goal)
    )
    
    # Generate obstacles
    print("Generating obstacles...")
    obstacles = planner.generate_obstacles(
        num_obstacles=args.num_obstacles,
        obstacle_types=args.obstacle_types,
        min_cube_size=args.min_cube_size,
        max_cube_size=args.max_cube_size,
        min_sphere_radius=args.min_sphere_radius,
        max_sphere_radius=args.max_sphere_radius,
        seed=args.seed
    )
    print(f"Generated {len(obstacles)} obstacles")
    
    # Compute convex hulls
    print("Computing convex hulls...")
    hulls = planner.collision_detector.get_obstacle_hulls()
    print(f"Computed {len(hulls)} convex hulls")
    
    # Plan path
    print("Planning path...")
    path = planner.plan_path(include_diagonals=not args.no_diagonals)
    
    if path:
        print(f"Path found! Length: {len(path)} nodes")
        print(f"Path distance: {planner.get_path_length():.2f} units")
        print()
        print("Path preview (first 5 and last 5 nodes):")
        for i, node in enumerate(path[:5]):
            print(f"  {i}: {node}")
        if len(path) > 10:
            print("  ...")
        for i, node in enumerate(path[-5:], start=len(path) - 5):
            print(f"  {i}: {node}")
    else:
        print("No path found!")
        print("The obstacles may be blocking all possible paths.")
        print("Try:")
        print("  - Reducing the number of obstacles")
        print("  - Increasing workspace size")
        print("  - Using a different random seed")
        return 1
    
    # Get statistics
    stats = planner.get_statistics()
    print()
    print("Statistics:")
    print(f"  Total obstacle points: {stats['total_obstacle_points']}")
    print(f"  Total hull faces: {stats['total_hull_faces']}")
    print(f"  Path length: {stats['path_length']:.2f} units")
    print(f"  Path nodes: {stats['path_num_nodes']}")
    
    # Export data if requested
    if args.export:
        print()
        print(f"Exporting data to {args.export}...")
        export_path_data(planner, args.export)
        print("Export complete!")
        
        if args.visualize:
            print()
            print("Starting visualization server...")
            print("Open http://localhost:8000/index.html in your browser")
            print("Load the exported JSON file to visualize the path")
            print("Press Ctrl+C to stop the server")
            print()
            
            # Start server
            import subprocess
            visualizer_dir = os.path.join(os.path.dirname(__file__), 'visualizer')
            server_script = os.path.join(visualizer_dir, 'start_server.py')
            subprocess.run([sys.executable, server_script])
    
    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

