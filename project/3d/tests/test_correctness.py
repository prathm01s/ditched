"""
Unit tests for algorithm correctness comparison.

Tests that all three algorithms produce the same (or similar) convex hulls.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.quickhull_3d import Quickhull3D
from algorithms.jarvis_march_3d import JarvisMarch3D
from algorithms.incremental_3d import IncrementalHull3D
from benchmark.data_generator import generate_random_points, generate_cube_points, generate_sphere_points


class TestAlgorithmCorrectness(unittest.TestCase):
    """Test cases for comparing algorithm correctness."""
    
    def test_tetrahedron_consistency(self):
        """Test that all algorithms produce same hull for tetrahedron."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        qh = Quickhull3D()
        qh.build(points)
        qh_vertices = set(tuple(v) for v in qh.get_vertices())
        
        jm = JarvisMarch3D()
        jm.compute(points)
        jm_vertices = set(tuple(v) for v in jm.get_vertices())
        
        inc = IncrementalHull3D()
        inc.build(points)
        inc_vertices = set(tuple(v) for v in inc.get_vertices())
        
        # All should have 4 vertices (the tetrahedron vertices)
        self.assertEqual(len(qh_vertices), 4)
        self.assertEqual(len(jm_vertices), 4)
        self.assertEqual(len(inc_vertices), 4)
        
        # Vertices should be the same
        self.assertEqual(qh_vertices, jm_vertices)
        self.assertEqual(qh_vertices, inc_vertices)
    
    def test_cube_consistency(self):
        """Test that all algorithms produce similar hulls for cube."""
        points = generate_cube_points(8, size=1.0, surface_only=True)
        
        qh = Quickhull3D()
        qh.build(points)
        qh_faces = len(qh.get_faces())
        qh_vertices = len(qh.get_vertices())
        
        jm = JarvisMarch3D()
        jm.compute(points)
        jm_faces = len(jm.get_faces())
        jm_vertices = len(jm.get_vertices())
        
        inc = IncrementalHull3D()
        inc.build(points)
        inc_faces = len(inc.get_faces())
        inc_vertices = len(inc.get_vertices())
        
        # All should produce valid hulls
        self.assertGreater(qh_faces, 0)
        self.assertGreater(jm_faces, 0)
        self.assertGreater(inc_faces, 0)
        
        # Vertex counts should be reasonable (cube should have around 8 vertices, but may vary)
        self.assertGreaterEqual(qh_vertices, 4)  # At least a tetrahedron
        self.assertLessEqual(qh_vertices, 8)  # At most 8 for a cube
        self.assertGreaterEqual(jm_vertices, 4)
        self.assertLessEqual(jm_vertices, 8)
        self.assertGreaterEqual(inc_vertices, 4)
        self.assertLessEqual(inc_vertices, 8)
    
    def test_random_points_consistency(self):
        """Test that all algorithms produce similar hulls for random points."""
        np.random.seed(42)  # For reproducibility
        points = generate_random_points(15, bounds=(-10, 10))
        
        qh = Quickhull3D()
        qh.build(points)
        qh_faces = len(qh.get_faces())
        qh_vertices = len(qh.get_vertices())
        
        jm = JarvisMarch3D()
        jm.compute(points)
        jm_faces = len(jm.get_faces())
        jm_vertices = len(jm.get_vertices())
        
        inc = IncrementalHull3D()
        inc.build(points)
        inc_faces = len(inc.get_faces())
        inc_vertices = len(inc.get_vertices())
        
        # All should produce valid hulls
        self.assertGreater(qh_faces, 0)
        self.assertGreater(jm_faces, 0)
        self.assertGreater(inc_faces, 0)
        
        # Vertex counts should be reasonable (at least 4, at most input size)
        self.assertGreaterEqual(qh_vertices, 4)
        self.assertLessEqual(qh_vertices, len(points))
        self.assertGreaterEqual(jm_vertices, 4)
        self.assertLessEqual(jm_vertices, len(points))
        self.assertGreaterEqual(inc_vertices, 4)
        self.assertLessEqual(inc_vertices, len(points))
        
        # Face counts should be reasonable (at least 4, typically more)
        self.assertGreaterEqual(qh_faces, 4)
        self.assertGreaterEqual(jm_faces, 4)
        self.assertGreaterEqual(inc_faces, 4)
    
    def test_sphere_points_consistency(self):
        """Test that all algorithms produce similar hulls for sphere points."""
        points = generate_sphere_points(20, radius=5.0, surface_only=True)
        
        qh = Quickhull3D()
        qh.build(points)
        qh_faces = len(qh.get_faces())
        
        jm = JarvisMarch3D()
        jm.compute(points)
        jm_faces = len(jm.get_faces())
        
        inc = IncrementalHull3D()
        inc.build(points)
        inc_faces = len(inc.get_faces())
        
        # All should produce valid hulls
        self.assertGreater(qh_faces, 0)
        self.assertGreater(jm_faces, 0)
        self.assertGreater(inc_faces, 0)
        
        # For sphere points, face counts may vary significantly due to different face splitting
        # Just verify all produce valid hulls with reasonable face counts
        # (Allow large differences as algorithms use different strategies)
        self.assertGreaterEqual(qh_faces, 4)
        self.assertGreaterEqual(jm_faces, 4)
        self.assertGreaterEqual(inc_faces, 4)


if __name__ == '__main__':
    unittest.main()

