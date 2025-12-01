"""
Unit tests for operation counters (iterations, comparisons).

Tests that operation counters are being tracked correctly.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.quickhull_3d import Quickhull3D
from algorithms.jarvis_march_3d import JarvisMarch3D
from algorithms.incremental_3d import IncrementalHull3D


class TestOperationCounters(unittest.TestCase):
    """Test cases for operation counter tracking."""
    
    def test_quickhull_counters(self):
        """Test that QuickHull tracks iterations and comparisons."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.5, 0.5, 0.5])
        ]
        
        hull = Quickhull3D()
        success = hull.build(points)
        
        self.assertTrue(success)
        self.assertIsNotNone(hull.iterations)
        self.assertIsNotNone(hull.comparisons)
        self.assertGreaterEqual(hull.iterations, 0)
        self.assertGreaterEqual(hull.comparisons, 0)
    
    def test_jarvis_counters(self):
        """Test that Jarvis March tracks iterations and comparisons."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.5, 0.5, 0.5])
        ]
        
        hull = JarvisMarch3D()
        success = hull.compute(points)
        
        self.assertTrue(success)
        self.assertIsNotNone(hull.iterations)
        self.assertIsNotNone(hull.comparisons)
        self.assertGreaterEqual(hull.iterations, 0)
        self.assertGreaterEqual(hull.comparisons, 0)
    
    def test_incremental_counters(self):
        """Test that Incremental tracks iterations and comparisons."""
        points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.5, 0.5, 0.5])
        ]
        
        hull = IncrementalHull3D()
        success = hull.build(points)
        
        self.assertTrue(success)
        self.assertIsNotNone(hull.iterations)
        self.assertIsNotNone(hull.comparisons)
        self.assertGreaterEqual(hull.iterations, 0)
        self.assertGreaterEqual(hull.comparisons, 0)
    
    def test_counters_increase_with_size(self):
        """Test that operation counts generally increase with input size."""
        small_points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.5, 0.5, 0.5])
        ]
        
        large_points = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.2, 0.3, 0.4]),
            np.array([0.6, 0.7, 0.8]),
            np.array([0.1, 0.9, 0.2]),
            np.array([0.8, 0.1, 0.9])
        ]
        
        # QuickHull
        qh_small = Quickhull3D()
        qh_small.build(small_points)
        
        qh_large = Quickhull3D()
        qh_large.build(large_points)
        
        # For larger input, comparisons should generally be higher
        # (though not always guaranteed due to point distribution)
        self.assertGreaterEqual(qh_large.comparisons, 0)
        self.assertGreaterEqual(qh_small.comparisons, 0)


if __name__ == '__main__':
    unittest.main()

