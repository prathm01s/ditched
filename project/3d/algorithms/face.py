"""
Face class for representing triangular faces in a 3D convex hull.

Each face is a triangle defined by three vertices and maintains connectivity
information through half-edges.
"""

from typing import List, Optional
import numpy as np
from .half_edge import HalfEdge, FaceTypes
from .geometry_utils import signed_area, signed_distance_to_plane, remove_vertex_from_list


class Face:
    """
    Represents a triangular face in the convex hull.
    
    Each face has three vertices, a normal vector, and maintains half-edge
    connectivity for mesh topology.
    """
    
    _id_counter = 0
    INTERSECTION_TOLERANCE = 0.01
    
    def __init__(self, step: int = 0):
        """
        Initialize a face.
        
        Args:
            step: The step number when this face was created (for animation tracking)
        """
        self.normal: np.ndarray = np.zeros(3)  # Face normal vector
        self.centroid: np.ndarray = np.zeros(3)  # Face centroid
        self.mark: int = FaceTypes.VISIBLE  # Face state (VISIBLE, NON_CONVEX, DELETED)
        self.points: List[Optional[np.ndarray]] = [None, None, None]  # Three vertices
        self.half_edges: List[Optional[HalfEdge]] = [None, None, None]  # Three half-edges
        self.outside: List[np.ndarray] = []  # Vertices outside this face
        self.created_at: int = step  # Step when face was created
        self.deleted_at: Optional[int] = None  # Step when face was deleted (if applicable)
        self.id = Face._id_counter
        Face._id_counter += 1
    
    def build_from_points(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """
        Build a face from three points.
        
        Args:
            a: numpy array of shape (3,) representing the first vertex
            b: numpy array of shape (3,) representing the second vertex
            c: numpy array of shape (3,) representing the third vertex
        """
        self.points = [a, b, c]
        self.normal = signed_area(a, b, c)
        norm = np.linalg.norm(self.normal)
        if norm > 1e-9:
            self.normal = self.normal / norm
        
        # Create half-edges for each vertex
        for i in range(3):
            self.half_edges[i] = HalfEdge(self.points[i], self)
        
        # Link half-edges in a cycle
        for i in range(2):
            self.half_edges[i].next = self.half_edges[i + 1]
        self.half_edges[2].next = self.half_edges[0]
        
        for i in range(1, 3):
            self.half_edges[i].prev = self.half_edges[i - 1]
        self.half_edges[0].prev = self.half_edges[2]
        
        # Update centroid
        self.centroid = (a + b + c) / 3.0
    
    def build_from_point_and_half_edge(self, p: np.ndarray, he: HalfEdge):
        """
        Build a face from a point and a half-edge.
        
        Args:
            p: numpy array of shape (3,) representing the new vertex
            he: HalfEdge representing an edge to connect to
        """
        self.points = [p, he.tail(), he.head()]
        self.normal = signed_area(self.points[0], self.points[1], self.points[2])
        norm = np.linalg.norm(self.normal)
        if norm > 1e-9:
            self.normal = self.normal / norm
        
        # Create half-edges
        self.half_edges[0] = HalfEdge(p, self)
        self.half_edges[1] = HalfEdge(he.tail(), self)
        self.half_edges[2] = he
        he.face = self
        
        # Link half-edges
        for i in range(2):
            self.half_edges[i].next = self.half_edges[i + 1]
        self.half_edges[2].next = self.half_edges[0]
        
        for i in range(1, 3):
            self.half_edges[i].prev = self.half_edges[i - 1]
        self.half_edges[0].prev = self.half_edges[2]
        
        # Update centroid
        self.centroid = (self.points[0] + self.points[1] + self.points[2]) / 3.0
    
    def signed_distance_from_point(self, p: np.ndarray) -> float:
        """
        Calculate the signed distance from a point to this face's plane.
        
        Args:
            p: numpy array of shape (3,) representing the point
            
        Returns:
            float: The signed distance (positive = on normal side)
        """
        return signed_distance_to_plane(self.points[0], self.points[1], self.points[2], p)
    
    def distance_to_point(self, p: np.ndarray) -> float:
        """
        Calculate the absolute distance from a point to this face's plane.
        
        Args:
            p: numpy array of shape (3,) representing the point
            
        Returns:
            float: The absolute distance
        """
        return abs(self.signed_distance_from_point(p))
    
    def has_empty_outside_set(self) -> bool:
        """
        Check if the outside set is empty.
        
        Returns:
            bool: True if no vertices are outside this face
        """
        return len(self.outside) == 0
    
    def mark_as_deleted(self, step: int):
        """
        Mark this face as deleted.
        
        Args:
            step: The step number when deletion occurred
        """
        self.mark = FaceTypes.DELETED
        self.deleted_at = step
    
    def remove_vertex_from_outside_set(self, remove: np.ndarray, epsilon: float = 1e-9):
        """
        Remove a vertex from the outside set.
        
        Args:
            remove: numpy array of shape (3,) representing the vertex to remove
            epsilon: Tolerance for equality comparison
        """
        remove_vertex_from_list(remove, self.outside, epsilon)
    
    def find_edge_with_extremities(self, a: np.ndarray, b: np.ndarray, epsilon: float = 1e-9) -> Optional[HalfEdge]:
        """
        Find the half-edge with given endpoints.
        
        Args:
            a: numpy array of shape (3,) representing the first endpoint
            b: numpy array of shape (3,) representing the second endpoint
            epsilon: Tolerance for equality comparison
            
        Returns:
            Optional HalfEdge: The half-edge with these endpoints, or None if not found
        """
        for he in self.half_edges:
            if he is None:
                continue
            head = he.head()
            tail = he.tail()
            if tail is None:
                continue
            
            equals_ab = np.allclose(head, a, atol=epsilon) and np.allclose(tail, b, atol=epsilon)
            equals_ba = np.allclose(head, b, atol=epsilon) and np.allclose(tail, a, atol=epsilon)
            
            if equals_ab or equals_ba:
                return he
        
        return None
    
    def equals_same_orientation(self, other: 'Face', epsilon: float = 1e-9) -> bool:
        """
        Check if this face equals another with the same orientation.
        
        Args:
            other: Another Face object to compare with
            epsilon: Tolerance for equality comparison
            
        Returns:
            bool: True if faces have the same vertices in the same order
        """
        for i in range(3):
            if not np.allclose(self.points[i], other.points[i], atol=epsilon):
                return False
        return True
    
    def equals_opposite_orientation(self, other: 'Face', epsilon: float = 1e-9) -> bool:
        """
        Check if this face equals another with opposite orientation.
        
        Args:
            other: Another Face object to compare with
            epsilon: Tolerance for equality comparison
            
        Returns:
            bool: True if faces have the same vertices in reverse order
        """
        for i in range(3):
            if not np.allclose(self.points[i], other.points[2 - i], atol=epsilon):
                return False
        return True
    
    def get_neighbor_face_ids(self) -> List[int]:
        """
        Get the IDs of neighboring faces.
        
        Returns:
            List[int]: List of face IDs that share an edge with this face
        """
        neighbor_ids = []
        for he in self.half_edges:
            if he is None or he.twin is None:
                continue
            opposite_face = he.opposite_face()
            if opposite_face is not None:
                neighbor_ids.append(opposite_face.id)
        return neighbor_ids

