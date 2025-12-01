"""
Unit tests for geometry utilities.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.geometry_utils import (
    point_to_line_distance,
    signed_distance_to_plane,
    cross_product,
    dot_product,
    vector_normalize,
    vector_length
)


class TestGeometryUtils(unittest.TestCase):
    """Test cases for geometry utility functions."""
    
    def test_point_to_line_distance(self):
        """Test point to line distance calculation."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        p = np.array([0, 1, 0])
        
        distance = point_to_line_distance(a, b, p)
        self.assertAlmostEqual(distance, 1.0, places=5)
    
    def test_signed_distance_to_plane(self):
        """Test signed distance to plane calculation."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 1, 0])
        p = np.array([0, 0, 1])
        
        distance = signed_distance_to_plane(a, b, c, p)
        self.assertGreater(distance, 0)  # Point should be on positive side
    
    def test_cross_product(self):
        """Test cross product calculation."""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        
        result = cross_product(a, b)
        expected = np.array([0, 0, 1])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_dot_product(self):
        """Test dot product calculation."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        result = dot_product(a, b)
        expected = 32  # 1*4 + 2*5 + 3*6
        
        self.assertAlmostEqual(result, expected)
    
    def test_vector_normalize(self):
        """Test vector normalization."""
        v = np.array([3, 4, 0])
        normalized = vector_normalize(v)
        
        length = vector_length(normalized)
        self.assertAlmostEqual(length, 1.0, places=5)


if __name__ == '__main__':
    unittest.main()

