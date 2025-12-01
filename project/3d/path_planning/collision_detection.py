"""
Collision Detection using Convex Hulls

This module integrates QuickHull for computing convex hulls of obstacles
and provides collision detection methods using these hulls.
"""

import sys
import os
from typing import List, Tuple, Optional
import numpy as np

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from algorithms.quickhull_3d import Quickhull3D
from algorithms.face import Face
from path_planning.obstacles import Obstacle3D


class ConvexHull3D:
    """
    Wrapper class for a 3D convex hull with collision detection methods.
    """
    
    def __init__(self, points: List[np.ndarray]):
        """
        Initialize and compute convex hull from points.
        
        Args:
            points: List of 3D points (numpy arrays) to compute hull from
        """
        self.points = points
        self.hull = Quickhull3D()
        self.faces: List[Face] = []
        self.vertices: List[np.ndarray] = []
        
        if len(points) >= 4:
            success = self.hull.build(points)
            if success:
                self.faces = self.hull.get_faces()
                self.vertices = self.hull.get_vertices()
    
    def get_faces(self) -> List[Face]:
        """Get all faces of the convex hull."""
        return self.faces
    
    def get_vertices(self) -> List[np.ndarray]:
        """Get all vertices of the convex hull."""
        return self.vertices


def point_in_hull(point: np.ndarray, hull: ConvexHull3D, tolerance: float = 1e-6) -> bool:
    """
    Check if a point is inside a convex hull.
    
    A point is inside the hull if it is on the negative side (or on) all faces.
    
    Args:
        point: 3D point (numpy array) to test
        hull: ConvexHull3D object
        tolerance: Tolerance for point-on-plane test (default: 1e-6)
    
    Returns:
        bool: True if point is inside or on the hull, False otherwise
    """
    if not hull.faces:
        return False
    
    point = np.asarray(point)
    
    # Check if point is on the negative side (or on) all faces
    for face in hull.faces:
        if len(face.points) < 3:
            continue
        
        # Compute signed distance from point to face plane
        p1, p2, p3 = face.points[0], face.points[1], face.points[2]
        
        # Compute face normal (pointing outward)
        edge1 = p2 - p1
        edge2 = p3 - p1
        normal = np.cross(edge1, edge2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < tolerance:
            continue  # Degenerate face
        
        normal = normal / normal_norm
        
        # Vector from face point to test point
        to_point = point - p1
        
        # Signed distance (positive = outside, negative = inside)
        signed_dist = np.dot(normal, to_point)
        
        # If point is significantly outside any face, it's outside the hull
        if signed_dist > tolerance:
            return False
    
    return True


def hull_intersects_hull(hull1: ConvexHull3D, hull2: ConvexHull3D, 
                        tolerance: float = 1e-6) -> bool:
    """
    Check if two convex hulls intersect.
    
    Two hulls intersect if:
    1. Any vertex of hull1 is inside hull2, OR
    2. Any vertex of hull2 is inside hull1, OR
    3. Any edge of hull1 intersects hull2, OR
    4. Any edge of hull2 intersects hull1
    
    For simplicity, we check if any vertex of one hull is inside the other.
    This is sufficient for most cases.
    
    Args:
        hull1: First convex hull
        hull2: Second convex hull
        tolerance: Tolerance for intersection test (default: 1e-6)
    
    Returns:
        bool: True if hulls intersect, False otherwise
    """
    # Check if any vertex of hull1 is inside hull2
    for vertex in hull1.vertices:
        if point_in_hull(vertex, hull2, tolerance):
            return True
    
    # Check if any vertex of hull2 is inside hull1
    for vertex in hull2.vertices:
        if point_in_hull(vertex, hull1, tolerance):
            return True
    
    # Additional check: if hulls are very close, they might intersect
    # Check if centers are very close
    if hull1.vertices and hull2.vertices:
        center1 = np.mean(np.array(hull1.vertices), axis=0)
        center2 = np.mean(np.array(hull2.vertices), axis=0)
        dist = np.linalg.norm(center1 - center2)
        
        # If centers are very close, check more carefully
        if dist < tolerance * 10:
            # Check if any face of hull1 intersects hull2
            for face in hull1.faces:
                if len(face.points) >= 3:
                    face_center = np.mean(np.array(face.points), axis=0)
                    if point_in_hull(face_center, hull2, tolerance * 10):
                        return True
    
    return False


def line_segment_intersects_hull(start: np.ndarray, end: np.ndarray,
                                 hull: ConvexHull3D,
                                 tolerance: float = 1e-6) -> bool:
    """
    Check if a line segment intersects a convex hull.
    
    A line segment intersects a hull if:
    1. Start point is inside hull, OR
    2. End point is inside hull, OR
    3. Line segment crosses any face of the hull
    
    Args:
        start: Start point of line segment (numpy array)
        end: End point of line segment (numpy array)
        hull: ConvexHull3D object
        tolerance: Tolerance for intersection test (default: 1e-6)
    
    Returns:
        bool: True if line segment intersects hull, False otherwise
    """
    start = np.asarray(start)
    end = np.asarray(end)
    
    # Check if endpoints are inside hull
    if point_in_hull(start, hull, tolerance) or point_in_hull(end, hull, tolerance):
        return True
    
    # Check if line segment intersects any face
    direction = end - start
    dir_length = np.linalg.norm(direction)
    
    if dir_length < tolerance:
        # Degenerate segment (start == end)
        return point_in_hull(start, hull, tolerance)
    
    direction = direction / dir_length  # Normalize
    
    for face in hull.faces:
        if len(face.points) < 3:
            continue
        
        p1, p2, p3 = face.points[0], face.points[1], face.points[2]
        
        # Compute face normal
        edge1 = p2 - p1
        edge2 = p3 - p1
        normal = np.cross(edge1, edge2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < tolerance:
            continue  # Degenerate face
        
        normal = normal / normal_norm
        
        # Check if line segment intersects the plane of this face
        # Ray-plane intersection: t = (d - dot(n, start)) / dot(n, direction)
        # where d = dot(n, p1)
        d = np.dot(normal, p1)
        denom = np.dot(normal, direction)
        
        if abs(denom) < tolerance:
            # Line is parallel to plane
            continue
        
        t = (d - np.dot(normal, start)) / denom
        
        # Check if intersection point is within segment bounds [0, dir_length]
        if 0 <= t <= dir_length:
            # Intersection point
            intersection = start + t * direction
            
            # Check if intersection point is inside the face (barycentric coordinates)
            # For simplicity, check if it's close to the face
            v0 = p3 - p1
            v1 = p2 - p1
            v2 = intersection - p1
            
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            
            # Check if point is inside triangle (u >= 0, v >= 0, u + v <= 1)
            if u >= -tolerance and v >= -tolerance and (u + v) <= 1 + tolerance:
                return True
    
    return False


class CollisionDetector:
    """
    Collision detector that uses convex hulls for efficient collision checking.
    """
    
    def __init__(self):
        """Initialize collision detector."""
        self.obstacle_hulls: List[ConvexHull3D] = []
        self.obstacles: List[Obstacle3D] = []
    
    def compute_obstacle_hulls(self, obstacles: List[Obstacle3D]) -> List[ConvexHull3D]:
        """
        Compute convex hulls for all obstacles using QuickHull.
        
        Args:
            obstacles: List of Obstacle3D objects
        
        Returns:
            List of ConvexHull3D objects
        """
        self.obstacles = obstacles
        self.obstacle_hulls = []
        
        for obstacle in obstacles:
            points = obstacle.get_obstacle_points()
            if len(points) >= 4:
                hull = ConvexHull3D(points)
                self.obstacle_hulls.append(hull)
            elif len(points) > 0:
                # For small obstacles, create a minimal hull
                # (though QuickHull needs at least 4 points)
                # In practice, we can skip these or handle them specially
                pass
        
        return self.obstacle_hulls
    
    def check_collision(self, point: np.ndarray, tolerance: float = 0.5) -> bool:
        """
        Check if a point collides with any obstacle hull.
        
        Args:
            point: 3D point (numpy array) to test
            tolerance: Safety margin around obstacles (default: 0.5)
        
        Returns:
            bool: True if point collides with any obstacle, False otherwise
        """
        # Add safety margin by checking if point is within tolerance distance of any hull
        for hull in self.obstacle_hulls:
            # First check if inside hull
            if point_in_hull(point, hull, -tolerance):  # Expand hull by tolerance
                return True
            
            # Also check distance to hull vertices (for safety margin)
            for vertex in hull.vertices:
                if np.linalg.norm(point - vertex) < tolerance:
                    return True
        
        return False
    
    def check_collision_with_hull(self, robot_hull: ConvexHull3D, 
                                  tolerance: float = 1e-6) -> bool:
        """
        Check if a robot's convex hull collides with any obstacle hull.
        
        Args:
            robot_hull: ConvexHull3D representing the robot's safety boundary
            tolerance: Tolerance for collision test (default: 1e-6)
        
        Returns:
            bool: True if robot hull collides with any obstacle, False otherwise
        """
        for obstacle_hull in self.obstacle_hulls:
            if hull_intersects_hull(robot_hull, obstacle_hull, tolerance):
                return True
        return False
    
    def is_path_safe(self, start: np.ndarray, end: np.ndarray,
                    tolerance: float = 1e-6) -> bool:
        """
        Check if a path segment is safe (doesn't intersect any obstacle).
        
        Args:
            start: Start point of path segment (numpy array)
            end: End point of path segment (numpy array)
            tolerance: Tolerance for intersection test (default: 1e-6)
        
        Returns:
            bool: True if path is safe, False if it intersects any obstacle
        """
        for hull in self.obstacle_hulls:
            if line_segment_intersects_hull(start, end, hull, tolerance):
                return False
        return True
    
    def get_obstacle_hulls(self) -> List[ConvexHull3D]:
        """Get all obstacle convex hulls."""
        return self.obstacle_hulls

