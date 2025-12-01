"""
Data generation utilities for 3D convex hull benchmarking.

This module provides functions to generate various types of 3D point datasets
for testing and benchmarking convex hull algorithms.
"""

from typing import List
import numpy as np
import random


def generate_random_points(n: int, bounds: tuple = (-10, 10)) -> List[np.ndarray]:
    """
    Generate n random 3D points within specified bounds.
    
    Args:
        n: Number of points to generate
        bounds: Tuple (min, max) for coordinate bounds (default: (-10, 10))
        
    Returns:
        List of numpy arrays of shape (3,) representing 3D points
    """
    points = []
    for _ in range(n):
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[0], bounds[1])
        z = random.uniform(bounds[0], bounds[1])
        points.append(np.array([x, y, z]))
    return points


def generate_sphere_points(n: int, radius: float = 10.0, 
                          surface_only: bool = True) -> List[np.ndarray]:
    """
    Generate points on or within a sphere.
    
    Args:
        n: Number of points to generate
        radius: Radius of the sphere (default: 10.0)
        surface_only: If True, generate points only on surface (default: True)
        
    Returns:
        List of numpy arrays of shape (3,) representing 3D points
    """
    points = []
    for _ in range(n):
        if surface_only:
            # Generate points uniformly on sphere surface
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
        else:
            # Generate points uniformly within sphere
            r = radius * (random.random() ** (1/3))
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
        points.append(np.array([x, y, z]))
    return points


def generate_cube_points(n: int, size: float = 10.0, 
                        surface_only: bool = False) -> List[np.ndarray]:
    """
    Generate points on or within a cube.
    
    Args:
        n: Number of points to generate
        size: Half-size of the cube (default: 10.0, so cube spans [-10, 10] in each dimension)
        surface_only: If True, generate points only on cube surface (default: False)
        
    Returns:
        List of numpy arrays of shape (3,) representing 3D points
    """
    points = []
    for _ in range(n):
        if surface_only:
            # Generate points on cube surface
            face = random.randint(0, 5)  # 6 faces
            if face == 0:  # x = -size
                x, y, z = -size, random.uniform(-size, size), random.uniform(-size, size)
            elif face == 1:  # x = size
                x, y, z = size, random.uniform(-size, size), random.uniform(-size, size)
            elif face == 2:  # y = -size
                x, y, z = random.uniform(-size, size), -size, random.uniform(-size, size)
            elif face == 3:  # y = size
                x, y, z = random.uniform(-size, size), size, random.uniform(-size, size)
            elif face == 4:  # z = -size
                x, y, z = random.uniform(-size, size), random.uniform(-size, size), -size
            else:  # z = size
                x, y, z = random.uniform(-size, size), random.uniform(-size, size), size
        else:
            # Generate points uniformly within cube
            x = random.uniform(-size, size)
            y = random.uniform(-size, size)
            z = random.uniform(-size, size)
        points.append(np.array([x, y, z]))
    return points


def generate_grid_points(n: int, bounds: tuple = (-10, 10)) -> List[np.ndarray]:
    """
    Generate points on a 3D grid.
    
    Args:
        n: Approximate number of points (will be rounded to nearest cube)
        bounds: Tuple (min, max) for coordinate bounds (default: (-10, 10))
        
    Returns:
        List of numpy arrays of shape (3,) representing 3D points
    """
    # Calculate grid size
    grid_size = int(np.ceil(n ** (1/3)))
    actual_n = grid_size ** 3
    
    points = []
    step = (bounds[1] - bounds[0]) / (grid_size - 1) if grid_size > 1 else 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = bounds[0] + i * step
                y = bounds[0] + j * step
                z = bounds[0] + k * step
                points.append(np.array([x, y, z]))
    
    return points


def load_points_from_file(filename: str) -> List[np.ndarray]:
    """
    Load 3D points from a text file.
    
    File format: one point per line, space-separated x y z coordinates.
    Lines starting with '#' are treated as comments and ignored.
    
    Args:
        filename: Path to the input file
        
    Returns:
        List of numpy arrays of shape (3,) representing 3D points
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    points.append(np.array([x, y, z]))
                except ValueError:
                    continue
    return points


def save_points_to_file(points: List[np.ndarray], filename: str):
    """
    Save 3D points to a text file.
    
    Args:
        points: List of numpy arrays of shape (3,) representing 3D points
        filename: Path to the output file
    """
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

