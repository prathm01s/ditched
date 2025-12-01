"""
Performance Comparison Utilities

This module provides functions to compare hull-based vs point-based
collision detection methods.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from path_planning.obstacles import Obstacle3D
from path_planning.collision_detection import CollisionDetector, point_in_hull, ConvexHull3D
from path_planning.workspace import Workspace3D
from path_planning.astar_3d import find_path


def point_in_obstacle_point_cloud(point: np.ndarray, obstacle: Obstacle3D, 
                                  tolerance: float = 1e-6) -> bool:
    """
    Check if a point is inside an obstacle using point-by-point comparison.
    
    This is a naive method that checks if the point is within the bounding box
    and then checks distance to obstacle points.
    
    Args:
        point: 3D point to test
        obstacle: Obstacle3D object
        tolerance: Tolerance for distance check
    
    Returns:
        bool: True if point is inside obstacle (approximate)
    """
    bounds = obstacle.get_bounds()
    
    # Check if point is within bounding box
    if not (bounds[0][0] <= point[0] <= bounds[0][1] and
            bounds[1][0] <= point[1] <= bounds[1][1] and
            bounds[2][0] <= point[2] <= bounds[2][1]):
        return False
    
    # For a more accurate check, we could compute distance to obstacle surface
    # For simplicity, we'll use a distance-based check
    obstacle_points = obstacle.get_obstacle_points()
    
    if obstacle.obstacle_type == 'sphere' and obstacle.center is not None:
        # For spheres, check distance to center
        dist_to_center = np.linalg.norm(point - obstacle.center)
        # Estimate radius from bounds
        radius = max(bounds[0][1] - bounds[0][0],
                    bounds[1][1] - bounds[1][0],
                    bounds[2][1] - bounds[2][0]) / 2.0
        return dist_to_center <= radius + tolerance
    else:
        # For cubes and other shapes, check if point is close to any obstacle point
        # This is a simplified check
        min_dist = min(np.linalg.norm(point - np.array(p)) for p in obstacle_points)
        return min_dist < tolerance * 10  # Approximate check


def check_collision_point_based(point: np.ndarray, obstacles: List[Obstacle3D],
                                tolerance: float = 1e-6) -> bool:
    """
    Check collision using point-by-point method.
    
    Args:
        point: 3D point to test
        obstacles: List of Obstacle3D objects
        tolerance: Tolerance for collision check
    
    Returns:
        bool: True if point collides with any obstacle
    """
    for obstacle in obstacles:
        if point_in_obstacle_point_cloud(point, obstacle, tolerance):
            return True
    return False


def benchmark_collision_methods(obstacles: List[Obstacle3D],
                                test_points: List[np.ndarray],
                                num_iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark hull-based vs point-based collision detection methods.
    
    Args:
        obstacles: List of Obstacle3D objects
        test_points: List of 3D points to test
        num_iterations: Number of iterations to run for averaging
    
    Returns:
        Dictionary containing performance metrics
    """
    # Setup hull-based collision detector
    collision_detector = CollisionDetector()
    collision_detector.compute_obstacle_hulls(obstacles)
    
    # Benchmark hull-based method
    hull_times = []
    hull_collision_checks = 0
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        for point in test_points:
            collision_detector.check_collision(point)
            hull_collision_checks += 1
        end_time = time.perf_counter()
        hull_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_hull_time = np.mean(hull_times)
    std_hull_time = np.std(hull_times)
    
    # Benchmark point-based method
    point_times = []
    point_collision_checks = 0
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        for point in test_points:
            check_collision_point_based(point, obstacles)
            point_collision_checks += 1
        end_time = time.perf_counter()
        point_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_point_time = np.mean(point_times)
    std_point_time = np.std(point_times)
    
    # Count collisions detected (should be similar for both methods)
    hull_collisions = sum(1 for point in test_points if collision_detector.check_collision(point))
    point_collisions = sum(1 for point in test_points if check_collision_point_based(point, obstacles))
    
    return {
        'hull_based': {
            'avg_time_ms': avg_hull_time,
            'std_time_ms': std_hull_time,
            'total_checks': hull_collision_checks,
            'collisions_detected': hull_collisions
        },
        'point_based': {
            'avg_time_ms': avg_point_time,
            'std_time_ms': std_point_time,
            'total_checks': point_collision_checks,
            'collisions_detected': point_collisions
        },
        'speedup': avg_point_time / avg_hull_time if avg_hull_time > 0 else 0.0,
        'num_obstacles': len(obstacles),
        'num_test_points': len(test_points)
    }


def compare_pathfinding_time(workspace: Workspace3D,
                            start: Tuple[int, int, int],
                            goal: Tuple[int, int, int],
                            obstacles: List[Obstacle3D],
                            num_iterations: int = 10) -> Dict[str, Any]:
    """
    Compare pathfinding time with hull-based vs point-based collision detection.
    
    Args:
        workspace: Workspace3D instance
        start: Start position
        goal: Goal position
        obstacles: List of Obstacle3D objects
        num_iterations: Number of iterations to run
    
    Returns:
        Dictionary containing performance metrics
    """
    # Setup hull-based collision detector
    hull_detector = CollisionDetector()
    hull_detector.compute_obstacle_hulls(obstacles)
    
    # Benchmark hull-based pathfinding
    hull_times = []
    hull_paths = []
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        path = find_path(workspace, start, goal, collision_detector=hull_detector)
        end_time = time.perf_counter()
        hull_times.append((end_time - start_time) * 1000)  # Convert to ms
        if path:
            hull_paths.append(path)
    
    avg_hull_time = np.mean(hull_times) if hull_times else 0.0
    hull_success_rate = len(hull_paths) / num_iterations if num_iterations > 0 else 0.0
    
    # For point-based, we'd need to implement a point-based collision detector
    # For now, we'll just return hull-based results
    # Point-based would be significantly slower for large obstacles
    
    return {
        'hull_based': {
            'avg_time_ms': avg_hull_time,
            'std_time_ms': np.std(hull_times) if hull_times else 0.0,
            'success_rate': hull_success_rate,
            'num_paths_found': len(hull_paths)
        },
        'num_obstacles': len(obstacles),
        'num_iterations': num_iterations
    }


def generate_test_points(workspace_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                        num_points: int = 100,
                        seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Generate random test points within workspace bounds.
    
    Args:
        workspace_bounds: Workspace bounds ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        num_points: Number of test points to generate
        seed: Random seed
    
    Returns:
        List of 3D points (numpy arrays)
    """
    if seed is not None:
        np.random.seed(seed)
    
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = workspace_bounds
    
    points = []
    for _ in range(num_points):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        points.append(np.array([x, y, z]))
    
    return points

