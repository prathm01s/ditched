"""
Robot Class with Safety Boundary

This module defines the Robot class which represents a robot with a safety boundary
modeled as a convex hull.
"""

from typing import Tuple, Optional
import numpy as np
from path_planning.collision_detection import ConvexHull3D, CollisionDetector


class Robot:
    """
    Robot class with position tracking and safety boundary as convex hull.
    
    The robot's safety boundary is a small cube or sphere around the robot's position,
    represented as a convex hull for collision detection.
    """
    
    def __init__(self, position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 safety_radius: float = 0.5):
        """
        Initialize a robot.
        
        Args:
            position: Initial position (x, y, z) of the robot
            safety_radius: Radius of the safety boundary (default: 0.5)
                          The robot is represented as a cube with side length 2 * safety_radius
        """
        self.position = np.array(position, dtype=float)
        self.safety_radius = safety_radius
        self.safety_hull: Optional[ConvexHull3D] = None
        self._update_safety_hull()
    
    def update_position(self, new_position: Tuple[float, float, float]):
        """
        Update the robot's position and recompute safety hull.
        
        Args:
            new_position: New position (x, y, z) of the robot
        """
        self.position = np.array(new_position, dtype=float)
        self._update_safety_hull()
    
    def _update_safety_hull(self):
        """Update the safety hull based on current position."""
        # Create a small cube around the robot's position
        # This represents the robot's safety boundary
        half_size = self.safety_radius
        
        # Generate 8 vertices of a cube centered at robot position
        vertices = [
            self.position + np.array([-half_size, -half_size, -half_size]),
            self.position + np.array([half_size, -half_size, -half_size]),
            self.position + np.array([half_size, half_size, -half_size]),
            self.position + np.array([-half_size, half_size, -half_size]),
            self.position + np.array([-half_size, -half_size, half_size]),
            self.position + np.array([half_size, -half_size, half_size]),
            self.position + np.array([half_size, half_size, half_size]),
            self.position + np.array([-half_size, half_size, half_size]),
        ]
        
        # For a cube, we need at least 8 points, but QuickHull needs 4+
        # Add some additional points on the faces for better hull representation
        # Actually, 8 vertices of a cube should be sufficient for QuickHull
        # But let's add a few more points on each face for robustness
        
        # Add points on each face (center of each face)
        face_centers = [
            self.position + np.array([0, 0, -half_size]),  # bottom face
            self.position + np.array([0, 0, half_size]),   # top face
            self.position + np.array([-half_size, 0, 0]),  # left face
            self.position + np.array([half_size, 0, 0]),   # right face
            self.position + np.array([0, -half_size, 0]),  # front face
            self.position + np.array([0, half_size, 0]),   # back face
        ]
        
        all_points = vertices + face_centers
        
        # Compute convex hull of safety boundary
        self.safety_hull = ConvexHull3D(all_points)
    
    def get_safety_hull(self) -> Optional[ConvexHull3D]:
        """
        Get the robot's safety boundary as a convex hull.
        
        Returns:
            ConvexHull3D object representing the safety boundary, or None if not computed
        """
        return self.safety_hull
    
    def get_position(self) -> np.ndarray:
        """
        Get the current position of the robot.
        
        Returns:
            numpy array representing the robot's position (x, y, z)
        """
        return self.position.copy()
    
    def check_collision_at_position(self, position: Tuple[float, float, float],
                                   collision_detector: CollisionDetector) -> bool:
        """
        Check if the robot would collide with obstacles at a given position.
        
        Args:
            position: Position (x, y, z) to check
            collision_detector: CollisionDetector instance with obstacle hulls
        
        Returns:
            bool: True if robot would collide at this position, False otherwise
        """
        # Temporarily update position
        old_position = self.position.copy()
        self.update_position(position)
        
        # Check collision
        safety_hull = self.get_safety_hull()
        if safety_hull is None:
            # Restore position
            self.position = old_position
            self._update_safety_hull()
            return True  # Assume collision if hull can't be computed
        
        collides = collision_detector.check_collision_with_hull(safety_hull)
        
        # Restore position
        self.position = old_position
        self._update_safety_hull()
        
        return collides
    
    def get_safety_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get the bounding box of the robot's safety boundary.
        
        Returns:
            Tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        """
        half_size = self.safety_radius
        x, y, z = self.position
        return (
            (x - half_size, x + half_size),
            (y - half_size, y + half_size),
            (z - half_size, z + half_size)
        )

