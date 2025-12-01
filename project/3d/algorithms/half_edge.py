"""
Half-edge data structure for representing mesh topology.

Half-edges are used to maintain connectivity information between faces
in the convex hull representation.
"""

from typing import Optional
import numpy as np


class FaceTypes:
    """Constants for face marking states."""
    VISIBLE = 0
    NON_CONVEX = 1
    DELETED = 2


class HalfEdge:
    """
    Represents a half-edge in a half-edge data structure.
    
    A half-edge is a directed edge that belongs to exactly one face.
    Each edge in the mesh is represented by two half-edges (twin edges).
    """
    
    _id_counter = 0
    
    def __init__(self, vertex: np.ndarray, face: Optional['Face'] = None):
        """
        Initialize a half-edge.
        
        Args:
            vertex: numpy array of shape (3,) representing the head vertex
            face: Optional Face object that this half-edge belongs to
        """
        self.vertex = vertex  # Head vertex of this half-edge
        self.face = face  # Face this half-edge belongs to
        self.next: Optional['HalfEdge'] = None  # Next half-edge in the face
        self.prev: Optional['HalfEdge'] = None  # Previous half-edge in the face
        self.twin: Optional['HalfEdge'] = None  # Twin half-edge (opposite direction)
        self.id = HalfEdge._id_counter
        HalfEdge._id_counter += 1
    
    def head(self) -> np.ndarray:
        """
        Get the head vertex of this half-edge.
        
        Returns:
            numpy array of shape (3,): The head vertex
        """
        return self.vertex
    
    def tail(self) -> Optional[np.ndarray]:
        """
        Get the tail vertex of this half-edge (head of previous edge).
        
        Returns:
            numpy array of shape (3,): The tail vertex, or None if prev is None
        """
        if self.prev is None:
            return None
        return self.prev.vertex
    
    def vector(self) -> Optional[np.ndarray]:
        """
        Get the vector from tail to head.
        
        Returns:
            numpy array of shape (3,): The edge vector, or None if tail is None
        """
        tail = self.tail()
        if tail is None:
            return None
        return self.head() - tail
    
    def length(self) -> float:
        """
        Calculate the length of this half-edge.
        
        Returns:
            float: The length of the edge, or -1 if tail is None
        """
        tail = self.tail()
        if tail is None:
            return -1.0
        return np.linalg.norm(self.head() - tail)
    
    def length_squared(self) -> float:
        """
        Calculate the squared length of this half-edge.
        
        Returns:
            float: The squared length of the edge, or -1 if tail is None
        """
        tail = self.tail()
        if tail is None:
            return -1.0
        diff = self.head() - tail
        return np.dot(diff, diff)
    
    def opposite_face(self) -> Optional['Face']:
        """
        Get the face on the opposite side of this edge.
        
        Returns:
            Optional Face: The face of the twin half-edge, or None if no twin
        """
        if self.twin is None:
            return None
        return self.twin.face
    
    def set_twin(self, twin: 'HalfEdge'):
        """
        Set the twin half-edge and establish bidirectional relationship.
        
        Args:
            twin: The twin half-edge to link with
        """
        self.twin = twin
        twin.twin = self

