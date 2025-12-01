"""
Unit tests for collision detection module.
"""

import unittest
import numpy as np
from path_planning.collision_detection import (
    ConvexHull3D, point_in_hull, hull_intersects_hull,
    line_segment_intersects_hull, CollisionDetector
)
from path_planning.obstacles import Obstacle3D, generate_cube_obstacle, generate_sphere_obstacle


class TestConvexHull3D(unittest.TestCase):
    """Test ConvexHull3D class."""
    
    def test_cube_hull(self):
        """Test convex hull computation for a cube."""
        # Create a simple cube
        obstacle = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        points = obstacle.get_obstacle_points()
        
        hull = ConvexHull3D(points)
        self.assertGreater(len(hull.get_faces()), 0)
        self.assertGreater(len(hull.get_vertices()), 0)
    
    def test_sphere_hull(self):
        """Test convex hull computation for a sphere."""
        obstacle = generate_sphere_obstacle((5, 5, 5), 2.0, resolution=10)
        points = obstacle.get_obstacle_points()
        
        hull = ConvexHull3D(points)
        self.assertGreater(len(hull.get_faces()), 0)
        self.assertGreater(len(hull.get_vertices()), 0)


class TestPointInHull(unittest.TestCase):
    """Test point-in-hull detection."""
    
    def test_point_inside_cube(self):
        """Test that a point inside a cube is detected."""
        obstacle = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        points = obstacle.get_obstacle_points()
        hull = ConvexHull3D(points)
        
        # Point at center should be inside
        center = np.array([5.0, 5.0, 5.0])
        self.assertTrue(point_in_hull(center, hull))
        
        # Point outside should not be inside
        outside = np.array([10.0, 10.0, 10.0])
        self.assertFalse(point_in_hull(outside, hull))
    
    def test_point_on_surface(self):
        """Test that a point on the surface is detected as inside."""
        obstacle = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        points = obstacle.get_obstacle_points()
        hull = ConvexHull3D(points)
        
        # Point on surface
        surface = np.array([6.0, 5.0, 5.0])  # On the +x face
        # Should be detected as inside (within tolerance)
        self.assertTrue(point_in_hull(surface, hull, tolerance=0.1))


class TestHullIntersection(unittest.TestCase):
    """Test hull-hull intersection detection."""
    
    def test_non_intersecting_hulls(self):
        """Test that non-intersecting hulls are detected correctly."""
        obstacle1 = generate_cube_obstacle((2, 2, 2), 1.0, resolution=4)
        obstacle2 = generate_cube_obstacle((10, 10, 10), 1.0, resolution=4)
        
        hull1 = ConvexHull3D(obstacle1.get_obstacle_points())
        hull2 = ConvexHull3D(obstacle2.get_obstacle_points())
        
        self.assertFalse(hull_intersects_hull(hull1, hull2))
    
    def test_intersecting_hulls(self):
        """Test that intersecting hulls are detected correctly."""
        obstacle1 = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        obstacle2 = generate_cube_obstacle((6, 6, 6), 2.0, resolution=4)
        
        hull1 = ConvexHull3D(obstacle1.get_obstacle_points())
        hull2 = ConvexHull3D(obstacle2.get_obstacle_points())
        
        # These should intersect
        self.assertTrue(hull_intersects_hull(hull1, hull2))


class TestLineSegmentIntersection(unittest.TestCase):
    """Test line segment-hull intersection detection."""
    
    def test_segment_through_hull(self):
        """Test that a segment passing through a hull is detected."""
        obstacle = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        points = obstacle.get_obstacle_points()
        hull = ConvexHull3D(points)
        
        # Segment from outside to outside, passing through
        start = np.array([0.0, 5.0, 5.0])
        end = np.array([10.0, 5.0, 5.0])
        
        self.assertTrue(line_segment_intersects_hull(start, end, hull))
    
    def test_segment_missing_hull(self):
        """Test that a segment missing a hull is detected correctly."""
        obstacle = generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        points = obstacle.get_obstacle_points()
        hull = ConvexHull3D(points)
        
        # Segment that doesn't intersect
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 1.0, 1.0])
        
        self.assertFalse(line_segment_intersects_hull(start, end, hull))


class TestCollisionDetector(unittest.TestCase):
    """Test CollisionDetector class."""
    
    def test_collision_detection(self):
        """Test collision detection with multiple obstacles."""
        obstacles = [
            generate_cube_obstacle((5, 5, 5), 2.0, resolution=4),
            generate_sphere_obstacle((10, 10, 10), 1.5, resolution=10)
        ]
        
        detector = CollisionDetector()
        detector.compute_obstacle_hulls(obstacles)
        
        # Point inside first obstacle
        point1 = np.array([5.0, 5.0, 5.0])
        self.assertTrue(detector.check_collision(point1))
        
        # Point outside all obstacles
        point2 = np.array([0.0, 0.0, 0.0])
        self.assertFalse(detector.check_collision(point2))
    
    def test_path_safety(self):
        """Test path safety checking."""
        obstacles = [
            generate_cube_obstacle((5, 5, 5), 2.0, resolution=4)
        ]
        
        detector = CollisionDetector()
        detector.compute_obstacle_hulls(obstacles)
        
        # Safe path (around obstacle)
        start1 = np.array([0.0, 0.0, 0.0])
        end1 = np.array([0.0, 0.0, 10.0])
        self.assertTrue(detector.is_path_safe(start1, end1))
        
        # Unsafe path (through obstacle)
        start2 = np.array([0.0, 5.0, 5.0])
        end2 = np.array([10.0, 5.0, 5.0])
        self.assertFalse(detector.is_path_safe(start2, end2))


if __name__ == '__main__':
    unittest.main()

