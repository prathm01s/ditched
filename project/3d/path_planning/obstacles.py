"""
3D Obstacle Generation

This module provides functions to generate random 3D obstacles (cubes and spheres)
for the path planning workspace. Each obstacle stores its points for convex hull computation.
"""

from typing import List, Tuple, Optional
import numpy as np
import random


class Obstacle3D:
    """
    Base class for 3D obstacles.
    
    Each obstacle stores its points which can be used to compute a convex hull.
    """
    
    def __init__(self, obstacle_type: str, points: List[np.ndarray]):
        """
        Initialize an obstacle.
        
        Args:
            obstacle_type: Type of obstacle ('cube' or 'sphere')
            points: List of 3D points (numpy arrays) that define the obstacle
        """
        self.obstacle_type = obstacle_type
        self.points = points
        self.center: Optional[np.ndarray] = None
        self.bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    
    def get_obstacle_points(self) -> List[np.ndarray]:
        """
        Get all points that define this obstacle.
        
        Returns:
            List of numpy arrays representing 3D points
        """
        return self.points
    
    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get the bounding box of the obstacle.
        
        Returns:
            Tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        """
        if self.bounds is not None:
            return self.bounds
        
        if not self.points:
            return ((0, 0), (0, 0), (0, 0))
        
        points_array = np.array(self.points)
        min_bounds = np.min(points_array, axis=0)
        max_bounds = np.max(points_array, axis=0)
        
        self.bounds = (
            (float(min_bounds[0]), float(max_bounds[0])),
            (float(min_bounds[1]), float(max_bounds[1])),
            (float(min_bounds[2]), float(max_bounds[2]))
        )
        
        # Compute center
        if self.center is None:
            self.center = np.array([
                (self.bounds[0][0] + self.bounds[0][1]) / 2.0,
                (self.bounds[1][0] + self.bounds[1][1]) / 2.0,
                (self.bounds[2][0] + self.bounds[2][1]) / 2.0
            ])
        
        return self.bounds


def generate_cube_obstacle(center: Tuple[float, float, float], 
                          size: float,
                          resolution: int = 4) -> Obstacle3D:  # Reduced for faster generation
    """
    Generate a cube obstacle with points on its surface.
    
    Args:
        center: (x, y, z) center of the cube
        size: Side length of the cube
        resolution: Number of points per edge (default: 8)
                    Higher resolution = more points = more accurate hull
    
    Returns:
        Obstacle3D object containing points on the cube surface
    """
    cx, cy, cz = center
    half_size = size / 2.0
    
    points = []
    
    # Generate points on all 6 faces of the cube
    # Face 1: x = cx - half_size (front face)
    for i in range(resolution):
        for j in range(resolution):
            y = cy - half_size + (i / (resolution - 1)) * size
            z = cz - half_size + (j / (resolution - 1)) * size
            points.append(np.array([cx - half_size, y, z]))
    
    # Face 2: x = cx + half_size (back face)
    for i in range(resolution):
        for j in range(resolution):
            y = cy - half_size + (i / (resolution - 1)) * size
            z = cz - half_size + (j / (resolution - 1)) * size
            points.append(np.array([cx + half_size, y, z]))
    
    # Face 3: y = cy - half_size (bottom face)
    for i in range(resolution):
        for j in range(resolution):
            x = cx - half_size + (i / (resolution - 1)) * size
            z = cz - half_size + (j / (resolution - 1)) * size
            points.append(np.array([x, cy - half_size, z]))
    
    # Face 4: y = cy + half_size (top face)
    for i in range(resolution):
        for j in range(resolution):
            x = cx - half_size + (i / (resolution - 1)) * size
            z = cz - half_size + (j / (resolution - 1)) * size
            points.append(np.array([x, cy + half_size, z]))
    
    # Face 5: z = cz - half_size (left face)
    for i in range(resolution):
        for j in range(resolution):
            x = cx - half_size + (i / (resolution - 1)) * size
            y = cy - half_size + (j / (resolution - 1)) * size
            points.append(np.array([x, y, cz - half_size]))
    
    # Face 6: z = cz + half_size (right face)
    for i in range(resolution):
        for j in range(resolution):
            x = cx - half_size + (i / (resolution - 1)) * size
            y = cy - half_size + (j / (resolution - 1)) * size
            points.append(np.array([x, y, cz + half_size]))
    
    obstacle = Obstacle3D('cube', points)
    obstacle.center = np.array(center)
    return obstacle


def generate_sphere_obstacle(center: Tuple[float, float, float],
                            radius: float,
                            resolution: int = 6) -> Obstacle3D:  # Reduced for faster generation
    """
    Generate a sphere obstacle with points on its surface.
    
    Args:
        center: (x, y, z) center of the sphere
        radius: Radius of the sphere
        resolution: Approximate number of points to generate (default: 20)
                    Higher resolution = more points = more accurate hull
    
    Returns:
        Obstacle3D object containing points on the sphere surface
    """
    cx, cy, cz = center
    points = []
    
    # Generate points on sphere surface using spherical coordinates
    # Use Fibonacci sphere algorithm for even distribution
    num_points = max(20, resolution * resolution // 2)  # Reduced for faster generation
    
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
    
    for i in range(num_points):
        y = 1 - (i / (num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        
        theta = golden_angle * i  # golden angle increment
        
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        
        # Scale by radius and translate to center
        point = np.array([
            cx + x * radius,
            cy + y * radius,
            cz + z * radius
        ])
        points.append(point)
    
    obstacle = Obstacle3D('sphere', points)
    obstacle.center = np.array(center)
    return obstacle


def generate_random_obstacles(num_obstacles: int = 8,
                             workspace_bounds: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = ((0, 20), (0, 20), (0, 20)),
                             obstacle_types: List[str] = None,
                             min_cube_size: float = 2.0,
                             max_cube_size: float = 5.0,
                             min_sphere_radius: float = 1.0,
                             max_sphere_radius: float = 3.0,
                             seed: Optional[int] = None) -> List[Obstacle3D]:
    """
    Generate random 3D obstacles in the workspace.
    
    Args:
        num_obstacles: Number of obstacles to generate (default: 8)
        workspace_bounds: Bounds of the workspace ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        obstacle_types: List of obstacle types to use (default: ['cube', 'sphere'])
                       Each obstacle will be randomly chosen from this list
        min_cube_size: Minimum cube side length (default: 2.0)
        max_cube_size: Maximum cube side length (default: 5.0)
        min_sphere_radius: Minimum sphere radius (default: 1.0)
        max_sphere_radius: Maximum sphere radius (default: 3.0)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List of Obstacle3D objects
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if obstacle_types is None:
        obstacle_types = ['cube', 'sphere']
    
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = workspace_bounds
    
    obstacles = []
    
    for _ in range(num_obstacles):
        obstacle_type = random.choice(obstacle_types)
        
        if obstacle_type == 'cube':
            # Generate random cube
            size = random.uniform(min_cube_size, max_cube_size)
            half_size = size / 2.0
            
            # Ensure cube fits within workspace
            center_x = random.uniform(min_x + half_size, max_x - half_size)
            center_y = random.uniform(min_y + half_size, max_y - half_size)
            center_z = random.uniform(min_z + half_size, max_z - half_size)
            
            obstacle = generate_cube_obstacle((center_x, center_y, center_z), size)
            obstacles.append(obstacle)
        
        elif obstacle_type == 'sphere':
            # Generate random sphere
            radius = random.uniform(min_sphere_radius, max_sphere_radius)
            
            # Ensure sphere fits within workspace
            center_x = random.uniform(min_x + radius, max_x - radius)
            center_y = random.uniform(min_y + radius, max_y - radius)
            center_z = random.uniform(min_z + radius, max_z - radius)
            
            obstacle = generate_sphere_obstacle((center_x, center_y, center_z), radius)
            obstacles.append(obstacle)
    
    return obstacles

