"""
Geometry utility functions for 3D convex hull algorithms.

This module provides helper functions for geometric computations such as
point-to-line distances, signed distances to planes, and vector operations.
"""

import numpy as np
from typing import Tuple, List


def point_to_line_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate the distance from a point to a line defined by two points.
    
    The line is defined by points a and b. The distance is computed using
    the cross product formula: |(p-a) × (b-a)| / |b-a|
    
    Args:
        a: numpy array of shape (3,) representing the first point on the line
        b: numpy array of shape (3,) representing the second point on the line
        p: numpy array of shape (3,) representing the point to measure distance from
        
    Returns:
        float: The perpendicular distance from point p to the line through a and b
    """
    direction = b - a
    a_minus_p = p - a
    cross_product = np.cross(a_minus_p, direction)
    return np.linalg.norm(cross_product) / np.linalg.norm(direction)


def signed_distance_to_plane(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> float:
    """
    Calculate the signed distance from a point to a plane defined by three points.
    
    The plane is defined by points a, b, and c. The sign indicates which side
    of the plane the point is on (positive = on the side of the normal).
    
    Args:
        a: numpy array of shape (3,) representing the first point on the plane
        b: numpy array of shape (3,) representing the second point on the plane
        c: numpy array of shape (3,) representing the third point on the plane
        p: numpy array of shape (3,) representing the point to measure distance from
        
    Returns:
        float: The signed distance from point p to the plane through a, b, and c
    """
    edge1 = b - a
    edge2 = c - a
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    to_point = p - a
    return np.dot(normal, to_point)


def signed_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate the signed area vector of a triangle.
    
    The signed area is computed as: (oa × ob + ob × oc + oc × oa) / 2
    where o is the centroid of the triangle.
    
    Args:
        a: numpy array of shape (3,) representing the first vertex
        b: numpy array of shape (3,) representing the second vertex
        c: numpy array of shape (3,) representing the third vertex
        
    Returns:
        numpy array of shape (3,): The signed area vector (normal vector)
    """
    o = (a + b + c) / 3.0
    oa = a - o
    ob = b - o
    oc = c - o
    
    S = (np.cross(oa, ob) + np.cross(ob, oc) + np.cross(oc, oa)) / 2.0
    return S


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the 3D cross product of two vectors.
    
    Args:
        a: numpy array of shape (3,) representing the first vector
        b: numpy array of shape (3,) representing the second vector
        
    Returns:
        numpy array of shape (3,): The cross product a × b
    """
    return np.cross(a, b)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.
    
    Args:
        a: numpy array of shape (3,) representing the first vector
        b: numpy array of shape (3,) representing the second vector
        
    Returns:
        float: The dot product a · b
    """
    return np.dot(a, b)


def vector_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Subtract two vectors.
    
    Args:
        a: numpy array of shape (3,) representing the first vector
        b: numpy array of shape (3,) representing the second vector
        
    Returns:
        numpy array of shape (3,): The result of a - b
    """
    return a - b


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two vectors.
    
    Args:
        a: numpy array of shape (3,) representing the first vector
        b: numpy array of shape (3,) representing the second vector
        
    Returns:
        numpy array of shape (3,): The result of a + b
    """
    return a + b


def vector_normalize(v: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        v: numpy array of shape (3,) representing the vector to normalize
        epsilon: Small value to avoid division by zero (default: 1e-9)
        
    Returns:
        numpy array of shape (3,): The normalized vector, or zero vector if input is too small
    """
    length = np.linalg.norm(v)
    if length <= epsilon:
        return np.zeros(3)
    return v / length


def vector_length(v: np.ndarray) -> float:
    """
    Calculate the length (magnitude) of a vector.
    
    Args:
        v: numpy array of shape (3,) representing the vector
        
    Returns:
        float: The length of the vector
    """
    return np.linalg.norm(v)


def remove_vertex_from_list(remove: np.ndarray, vertex_list: List[np.ndarray], epsilon: float = 1e-9) -> bool:
    """
    Remove a vertex from a list if it matches (within epsilon tolerance).
    
    Args:
        remove: numpy array of shape (3,) representing the vertex to remove
        vertex_list: List of numpy arrays representing vertices
        epsilon: Tolerance for equality comparison (default: 1e-9)
        
    Returns:
        bool: True if vertex was found and removed, False otherwise
    """
    remove = np.asarray(remove)
    for i in range(len(vertex_list) - 1, -1, -1):  # Iterate backwards to avoid index issues
        vertex = np.asarray(vertex_list[i])
        if vertex.shape == remove.shape and np.allclose(vertex, remove, atol=epsilon):
            vertex_list.pop(i)
            return True
    return False

