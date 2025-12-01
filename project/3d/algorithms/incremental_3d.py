"""
Incremental 3D Convex Hull Algorithm Implementation.

This module implements the incremental algorithm for computing the 3D convex hull
of a set of points. The algorithm starts with an initial tetrahedron and adds
points one by one, updating the hull at each step.

Time Complexity: O(n²) average and worst case
Space Complexity: O(n)
"""

from typing import List, Optional, Set, Tuple
import numpy as np
from .face import Face, FaceTypes
from .geometry_utils import signed_distance_to_plane


class IncrementalHull3D:
    """
    Incremental 3D convex hull algorithm implementation.
    
    This class implements the incremental algorithm which builds the convex hull
    by starting with a tetrahedron and iteratively adding points, removing visible
    faces and creating new ones.
    """
    
    def __init__(self):
        """
        Initialize an IncrementalHull3D instance.
        
        Sets up data structures for the incremental algorithm:
        - vertex_list: Input points to compute hull for
        - faces: All faces in the current hull (including deleted ones)
        - epsilon: Numerical tolerance for floating-point comparisons
        - total_steps: Total number of algorithm steps
        - used_in_tetrahedron: Indices of points used in initial tetrahedron
        - iterations: Counter for algorithm iterations (for benchmarking)
        - comparisons: Counter for comparison operations (for benchmarking)
        """
        self.vertex_list: List[np.ndarray] = []  # Input points to compute hull for
        self.faces: List[Face] = []  # All faces in the hull (including deleted)
        self.epsilon: float = 1e-9  # Numerical tolerance for floating-point comparisons
        self.total_steps: int = 0  # Total number of steps
        self.used_in_tetrahedron: Set[int] = set()  # Indices of points in initial tetrahedron
        self.iterations: int = 0  # Counter for algorithm iterations (for benchmarking)
        self.comparisons: int = 0  # Counter for comparison operations (for benchmarking)
    
    def _are_non_coplanar(self, i: int, j: int, k: int, l: int) -> bool:
        """
        Check if 4 points are non-coplanar using volume test.
        
        Four points are coplanar if the volume of the tetrahedron they form is zero.
        We compute the signed volume using the scalar triple product and check if
        it's above the epsilon threshold.
        
        Args:
            i, j, k, l: Indices of points to check in vertex_list
            
        Returns:
            bool: True if points are non-coplanar (form a valid tetrahedron)
        """
        v0 = self.vertex_list[i]
        v1 = self.vertex_list[j]
        v2 = self.vertex_list[k]
        v3 = self.vertex_list[l]
        
        # Compute edge vectors from v0
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0
        
        # Volume = |(edge1 × edge2) · edge3| / 6
        # We skip the division by 6 since we only need to check if it's non-zero
        cross = np.cross(edge1, edge2)
        volume = abs(np.dot(cross, edge3))
        
        # Points are non-coplanar if volume is above threshold
        return volume > self.epsilon
    
    def _initialize_tetrahedron(self, step: int) -> bool:
        """
        Initialize the convex hull with the first 4 non-coplanar points.
        
        Searches for 4 points that form a valid tetrahedron (non-coplanar).
        Once found, creates 4 triangular faces with correct orientation
        (normals pointing outward).
        
        Args:
            step: Current step number for face creation tracking
            
        Returns:
            bool: True if successful, False if no 4 non-coplanar points found
        """
        # Limit search to first 10 points for efficiency
        # If we can't find 4 non-coplanar points in the first 10, the input is degenerate
        max_search = min(10, len(self.vertex_list))
        
        # Brute force search for 4 non-coplanar points
        for i in range(max_search):
            for j in range(i + 1, max_search):
                for k in range(j + 1, max_search):
                    for l in range(k + 1, max_search):
                        if self._are_non_coplanar(i, j, k, l):
                            p0 = self.vertex_list[i]
                            p1 = self.vertex_list[j]
                            p2 = self.vertex_list[k]
                            p3 = self.vertex_list[l]
                            
                            # Compute centroid of tetrahedron (for orientation check)
                            center = (p0 + p1 + p2 + p3) / 4.0
                            
                            # Helper function to create face with correct orientation
                            # Face normal should point outward (away from center)
                            def create_oriented_face(a, b, c):
                                """Create a face with normal pointing outward."""
                                f = Face(step)
                                f.build_from_points(a, b, c)
                                
                                face_point = a
                                to_center = center - face_point
                                
                                # If normal points toward center, flip face orientation
                                if np.dot(f.normal, to_center) > 0:
                                    f.build_from_points(a, c, b)
                                
                                return f
                            
                            # Create 4 faces of the tetrahedron
                            face1 = create_oriented_face(p0, p1, p2)  # opposite to p3
                            face2 = create_oriented_face(p0, p1, p3)  # opposite to p2
                            face3 = create_oriented_face(p0, p2, p3)  # opposite to p1
                            face4 = create_oriented_face(p1, p2, p3)  # opposite to p0
                            
                            self.faces = [face1, face2, face3, face4]
                            
                            # Track which points were used in the initial tetrahedron
                            self.used_in_tetrahedron = {i, j, k, l}
                            
                            return True
        
        return False
    
    def _find_horizon_edges(self, visible_face_indices: List[int],
                           active_faces: List[Face],
                           active_face_indices: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Find horizon edges from visible faces.
        
        The horizon is the boundary between visible and non-visible faces.
        An edge is a horizon edge if exactly one of its two adjacent faces is visible.
        These edges form a closed loop that will be connected to the new point.
        
        Args:
            visible_face_indices: List of indices of visible faces (to be deleted)
            active_faces: List of active (non-deleted) faces
            active_face_indices: Original indices of active faces in self.faces
            
        Returns:
            List of tuples: Horizon edges as [(v0, v1), ...] pairs with correct orientation
        """
        visible_set = set(visible_face_indices)
        edge_count = {}  # Count how many visible faces share each edge
        edge_to_non_visible_face = {}  # Map edge to non-visible face (for orientation)
        
        def normalize_edge(v1, v2):
            """Create a consistent key for edge matching."""
            k1 = f"{int(v1[0]*1e6)},{int(v1[1]*1e6)},{int(v1[2]*1e6)}"
            k2 = f"{int(v2[0]*1e6)},{int(v2[1]*1e6)},{int(v2[2]*1e6)}"
            key = f"{k1}|{k2}" if k1 < k2 else f"{k2}|{k1}"
            return key, (v1, v2)
        
        # First pass: collect edges from visible faces
        for i in range(len(active_faces)):
            face_idx = active_face_indices[i]
            if face_idx not in visible_set:
                continue
            
            face = active_faces[i]
            v = face.points
            
            edges = [
                (v[0], v[1]),
                (v[1], v[2]),
                (v[2], v[0])
            ]
            
            for edge in edges:
                key, _ = normalize_edge(edge[0], edge[1])
                edge_count[key] = edge_count.get(key, 0) + 1
        
        # Second pass: find edges from non-visible faces shared with visible faces
        for i in range(len(active_faces)):
            face_idx = active_face_indices[i]
            if face_idx in visible_set:
                continue
            
            face = active_faces[i]
            v = face.points
            
            edges = [
                (v[0], v[1]),
                (v[1], v[2]),
                (v[2], v[0])
            ]
            
            for edge in edges:
                key, original = normalize_edge(edge[0], edge[1])
                if key in edge_count:
                    # This edge is shared between visible and non-visible faces
                    edge_to_non_visible_face[key] = original
        
        # Collect horizon edges with correct orientation
        horizon = []
        for key, original_edge in edge_to_non_visible_face.items():
            # Return edge in correct orientation (reversed for outward normal)
            horizon.append((original_edge[1], original_edge[0]))
        
        return horizon
    
    def _add_point(self, point_idx: int, step: int):
        """
        Add a single point to the existing convex hull.
        
        This is the core operation of the incremental algorithm:
        1. Determine which faces are visible from the new point
        2. Find the horizon (boundary between visible and non-visible faces)
        3. Delete visible faces
        4. Create new faces connecting the point to horizon edges
        
        Args:
            point_idx: Index of the point to add in vertex_list
            step: Current step number for tracking face creation/deletion
        """
        point = self.vertex_list[point_idx]
        
        # Get only active (non-deleted) faces
        # We need to work with active faces only, but track original indices
        active_faces = []
        active_face_indices = []
        for face_idx in range(len(self.faces)):
            if self.faces[face_idx].mark != FaceTypes.DELETED:
                active_faces.append(self.faces[face_idx])
                active_face_indices.append(face_idx)
        
        # Step 1: Determine visibility of each face
        # A face is visible if the point is on the positive side (outside) of it
        visible_faces = []
        for i in range(len(active_faces)):
            face = active_faces[i]
            
            # Compute signed distance from point to face plane
            # Positive distance means point is outside (face is visible)
            distance = face.signed_distance_from_point(point)
            self.comparisons += 1
            
            if distance > self.epsilon:
                # Point is on the positive side (visible)
                visible_faces.append(active_face_indices[i])
        
        # If point is inside the hull, don't add it (no visible faces)
        if len(visible_faces) == 0:
            return
        
        # Special case: if point sees ALL active faces, the hull is invalid
        # This shouldn't happen in a valid incremental algorithm, but handle it gracefully
        if len(visible_faces) == len(active_faces):
            return
        
        # Step 2: Find horizon edges
        # Horizon edges are edges between visible and non-visible faces
        horizon_edges = self._find_horizon_edges(visible_faces, active_faces, active_face_indices)
        
        if len(horizon_edges) == 0:
            return
        
        # Step 3: Mark visible faces as deleted
        # We don't actually remove them from the list, just mark them
        for face_idx in visible_faces:
            self.faces[face_idx].mark_as_deleted(step)
        
        # Step 4: Add new faces connecting point to horizon
        # Each horizon edge becomes an edge of a new triangular face
        for edge in horizon_edges:
            new_face = Face(step)
            new_face.build_from_points(edge[0], edge[1], point)
            self.faces.append(new_face)
        
        self.iterations += 1
    
    def build(self, input_points: List[np.ndarray]) -> bool:
        """
        Build the convex hull using the incremental algorithm.
        
        The algorithm works by:
        1. Starting with an initial tetrahedron from 4 non-coplanar points
        2. Adding remaining points one by one
        3. For each point, removing visible faces and creating new ones
        
        Args:
            input_points: List of numpy arrays of shape (3,) representing 3D points
            
        Returns:
            bool: True if hull was built successfully, False otherwise
        """
        self.vertex_list = [np.array(p) for p in input_points]
        
        # Need at least 4 points to form a 3D convex hull (tetrahedron)
        if len(self.vertex_list) < 4:
            return False
        
        # Initialize algorithm state
        step = 0
        self.faces = []
        self.used_in_tetrahedron = set()
        self.iterations = 0
        self.comparisons = 0
        
        # Step 1: Initialize with a tetrahedron from first 4 non-coplanar points
        if not self._initialize_tetrahedron(step):
            return False
        step += 1
        
        # Step 2: Add remaining points one by one
        # Points used in the initial tetrahedron are already on the hull
        for i in range(len(self.vertex_list)):
            if i not in self.used_in_tetrahedron:
                self._add_point(i, step)
                step += 1
        
        self.total_steps = step
        return True
    
    def get_faces(self) -> List[Face]:
        """
        Get all active (non-deleted) faces in the hull.
        
        Returns:
            List[Face]: List of active Face objects
        """
        return [f for f in self.faces if f.mark != FaceTypes.DELETED]
    
    def get_vertices(self) -> List[np.ndarray]:
        """
        Get all unique vertices in the hull.
        
        Returns:
            List[np.ndarray]: List of unique vertex arrays
        """
        vertices = set()
        for face in self.get_faces():
            for point in face.points:
                vertices.add(tuple(point))
        return [np.array(v) for v in vertices]

