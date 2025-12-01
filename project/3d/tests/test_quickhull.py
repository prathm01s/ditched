"""
Unit tests for QuickHull 3D algorithm.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.quickhull_3d import Quickhull3D
from benchmark.data_generator import generate_cube_points, generate_sphere_points


class TestQuickHull3D(unittest.TestCase):
    """Test cases for QuickHull 3D algorithm."""
    
    def test_tetrahedron(self):
        """Test with a simple tetrahedron (4 points)."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        hull = Quickhull3D()
        success = hull.build(points)
        
        self.assertTrue(success)
        faces = hull.get_faces()
        self.assertEqual(len(faces), 4)  # Tetrahedron has 4 faces
    
    def test_cube(self):
        """Test with cube vertices."""
        points = generate_cube_points(8, size=1.0, surface_only=True)
        
        hull = Quickhull3D()
        success = hull.build(points)
        
        self.assertTrue(success)
        faces = hull.get_faces()
        self.assertGreater(len(faces), 0)
    
    def test_minimum_points(self):
        """Test that algorithm requires at least 4 points."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0])
        ]
        
        hull = Quickhull3D()
        success = hull.build(points)
        
        self.assertFalse(success)
    
    def test_sphere_points(self):
        """Test with points on sphere surface."""
        points = generate_sphere_points(20, radius=5.0, surface_only=True)
        
        hull = Quickhull3D()
        success = hull.build(points)
        
        self.assertTrue(success)
        faces = hull.get_faces()
        self.assertGreater(len(faces), 0)
        vertices = hull.get_vertices()
        self.assertGreater(len(vertices), 0)


if __name__ == '__main__':
    unittest.main()

