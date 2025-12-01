"""
Unit tests for edge cases and special scenarios.

Tests coplanar points, collinear points, duplicate points, etc.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.quickhull_3d import Quickhull3D
from algorithms.jarvis_march_3d import JarvisMarch3D
from algorithms.incremental_3d import IncrementalHull3D


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and special scenarios."""
    
    def test_coplanar_points(self):
        """Test with nearly coplanar points."""
        # Points nearly on a plane
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0.5, 0.5, 0.001]),  # Slightly off plane
            np.array([0.2, 0.3, 0.0005])
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        self.assertGreater(len(qh.get_faces()), 0)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        self.assertGreater(len(jm.get_faces()), 0)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)
        self.assertGreater(len(inc.get_faces()), 0)
    
    def test_duplicate_points(self):
        """Test with duplicate points."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 0]),  # Duplicate
            np.array([1, 0, 0])   # Duplicate
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)
    
    def test_exactly_four_points(self):
        """Test with exactly 4 points (minimum required)."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        self.assertEqual(len(qh.get_faces()), 4)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        self.assertGreater(len(jm.get_faces()), 0)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)
        self.assertEqual(len(inc.get_faces()), 4)
    
    def test_octahedron(self):
        """Test with octahedron vertices."""
        points = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1])
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        self.assertGreater(len(qh.get_faces()), 0)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        self.assertGreater(len(jm.get_faces()), 0)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)
        self.assertGreater(len(inc.get_faces()), 0)
    
    def test_large_coordinates(self):
        """Test with large coordinate values."""
        points = [
            np.array([1000, 0, 0]),
            np.array([-1000, 0, 0]),
            np.array([0, 1000, 0]),
            np.array([0, -1000, 0]),
            np.array([0, 0, 1000]),
            np.array([0, 0, -1000])
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)
    
    def test_small_coordinates(self):
        """Test with very small coordinate values."""
        points = [
            np.array([0.001, 0, 0]),
            np.array([-0.001, 0, 0]),
            np.array([0, 0.001, 0]),
            np.array([0, -0.001, 0]),
            np.array([0, 0, 0.001]),
            np.array([0, 0, -0.001])
        ]
        
        qh = Quickhull3D()
        success = qh.build(points)
        self.assertTrue(success)
        
        jm = JarvisMarch3D()
        success = jm.compute(points)
        self.assertTrue(success)
        
        inc = IncrementalHull3D()
        success = inc.build(points)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()

