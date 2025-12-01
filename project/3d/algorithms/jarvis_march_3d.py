"""
Jarvis March (Gift Wrapping) 3D Convex Hull Algorithm Implementation.

This module implements the Jarvis March algorithm (also known as gift wrapping)
for computing the 3D convex hull of a set of points. The algorithm wraps around
the points like wrapping a gift.

Time Complexity: O(nh) where n is number of points and h is number of hull vertices
Space Complexity: O(n)
"""

from typing import List, Optional, Set, Tuple
import numpy as np
from .face import Face
from .geometry_utils import (
    cross_product,
    dot_product,
    vector_subtract,
    vector_add,
    vector_normalize,
    vector_length
)


class JarvisMarch3D:
    """
    Jarvis March 3D convex hull algorithm implementation.
    
    This class implements the gift wrapping algorithm which builds the convex hull
    by starting with a bottommost point and iteratively finding the next face
    using dihedral angle calculations.
    """
    
    def __init__(self):
        """
        Initialize a JarvisMarch3D instance.
        
        Sets up data structures and counters needed for the algorithm:
        - vertex_list: Input points to compute hull for
        - faces: Triangular faces forming the convex hull
        - epsilon: Numerical tolerance for floating-point comparisons
        - iterations: Counter for algorithm iterations (for benchmarking)
        - comparisons: Counter for comparison operations (for benchmarking)
        """
        self.vertex_list: List[np.ndarray] = []  # Input points to compute hull for
        self.faces: List[Face] = []  # Triangular faces forming the convex hull
        self.epsilon: float = 1e-9  # Numerical tolerance for floating-point comparisons
        self.iterations: int = 0  # Counter for algorithm iterations (for benchmarking)
        self.comparisons: int = 0  # Counter for comparison operations (for benchmarking)
    
    def _vec_key(self, v: np.ndarray) -> str:
        """
        Create a string key for a vector (for hashing).
        
        Converts 3D coordinates to a formatted string with 6 decimal places.
        NumPy arrays aren't hashable, so we need string keys for use in sets/dicts.
        Used throughout the algorithm to track unique vertices and faces.
        
        Args:
            v: numpy array of shape (3,) representing a 3D point
            
        Returns:
            str: String representation of the vector (e.g., "1.234567,2.345678,3.456789")
        """
        # Format with 6 decimal places to balance precision and uniqueness
        return f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}"
    
    def _vec_equals(self, a: np.ndarray, b: np.ndarray, eps: float = None) -> bool:
        """
        Check if two vectors are equal within tolerance.
        
        Args:
            a: numpy array of shape (3,)
            b: numpy array of shape (3,)
            eps: Tolerance (defaults to self.epsilon)
            
        Returns:
            bool: True if vectors are equal within tolerance
        """
        if eps is None:
            eps = self.epsilon
        return np.allclose(a, b, atol=eps)
    
    def _signed_volume(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
        """
        Calculate the signed volume of a tetrahedron.
        
        Uses the scalar triple product formula: (ab × ac) · ad
        The sign indicates which side of plane abc point d is on (positive = outside).
        Used to determine point orientation and face validity.
        
        Args:
            a, b, c: numpy arrays of shape (3,) representing three points defining a plane
            d: numpy array of shape (3,) representing the point to test
            
        Returns:
            float: Signed volume (positive if d is on positive side of plane abc)
        """
        # Compute edge vectors from point a
        ab = b - a
        ac = c - a
        ad = d - a
        # Cross product gives normal to plane abc, dot with ad gives signed volume
        cross_ab_ac = cross_product(ab, ac)
        return dot_product(cross_ab_ac, ad)
    
    def find_bottommost_point(self) -> int:
        """
        Find the bottommost point using lexicographic ordering.
        
        The bottommost point is defined as the point with:
        1. Lowest z-coordinate
        2. If z is equal, lowest y-coordinate
        3. If both z and y are equal, lowest x-coordinate
        
        This point is guaranteed to be on the convex hull and serves as a starting
        point for the gift wrapping algorithm.
        
        Returns:
            int: Index of the bottommost point in vertex_list
        """
        bottom_idx = 0
        points = self.vertex_list
        
        # Iterate through all points to find the bottommost one
        for i in range(1, len(points)):
            p1 = points[bottom_idx]
            p2 = points[i]
            
            # Lexicographic comparison: z first, then y, then x
            if (p2[2] < p1[2] or
                (abs(p2[2] - p1[2]) < self.epsilon and p2[1] < p1[1]) or
                (abs(p2[2] - p1[2]) < self.epsilon and
                 abs(p2[1] - p1[1]) < self.epsilon and
                 p2[0] < p1[0])):
                bottom_idx = i
            # Track comparison operations for benchmarking
            self.comparisons += 3
        
        return bottom_idx
    
    def is_valid_hull_face(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """
        Check if a face is a valid hull face (all points on one side of the plane).
        
        A face is valid if all other points (not already on the hull) are on the
        same side of the plane defined by the face. This ensures the face is part
        of the convex hull boundary.
        
        Args:
            p1, p2, p3: numpy arrays of shape (3,) representing face vertices
            
        Returns:
            bool: True if face is valid (all other points on one side), False otherwise
        """
        # Collect all vertices already on the hull to skip them
        hull_vertices = set()
        for face in self.faces:
            if face.points and len(face.points) >= 3:
                for pt in face.points:
                    hull_vertices.add(self._vec_key(pt))
        
        # Add the candidate face vertices to the set
        hull_vertices.add(self._vec_key(p1))
        hull_vertices.add(self._vec_key(p2))
        hull_vertices.add(self._vec_key(p3))
        
        # Check all points not on the hull
        for p in self.vertex_list:
            key = self._vec_key(p)
            # Skip points already on the hull
            if key in hull_vertices:
                continue
            
            # Check which side of the plane this point is on
            vol = self._signed_volume(p1, p2, p3, p)
            # Skip coplanar points (within epsilon)
            if abs(vol) <= self.epsilon:
                continue
            # If point is on the negative side, face is not valid
            if vol < -self.epsilon:
                return False
            self.comparisons += 1
        
        return True
    
    def find_next_face_point(self, edge_p1: np.ndarray, edge_p2: np.ndarray,
                            current_face_normal: Optional[np.ndarray] = None) -> int:
        """
        Find the next point to form a face with an edge using dihedral angle.
        
        This is the core of the gift wrapping algorithm. For a given edge, we find
        the point that forms the smallest dihedral angle, which ensures we're
        wrapping around the hull correctly. The dihedral angle is measured between
        the current face and the candidate face.
        
        Args:
            edge_p1: numpy array of shape (3,) representing first edge endpoint
            edge_p2: numpy array of shape (3,) representing second edge endpoint
            current_face_normal: Optional normal vector of current face (for angle calculation)
            
        Returns:
            int: Index of the best point in vertex_list, or -1 if none found
        """
        best_idx = -1
        min_angle = np.inf
        
        # Normalize the edge vector for angle calculations
        edge_vec = edge_p2 - edge_p1
        edge_norm = vector_normalize(edge_vec, self.epsilon)
        
        # Compute reference perpendicular vector for angle measurement
        # This vector is perpendicular to the edge and lies in the current face plane
        ref_perp = None
        if current_face_normal is not None and vector_length(current_face_normal) > self.epsilon:
            normal_norm = vector_normalize(current_face_normal, self.epsilon)
            # Project normal onto edge to get component perpendicular to edge
            dot_edge = dot_product(normal_norm, edge_norm)
            ref_perp = vector_normalize(normal_norm - edge_norm * dot_edge, self.epsilon)
        
        # Try each point as a candidate for the next face
        for i in range(len(self.vertex_list)):
            p = self.vertex_list[i]
            # Skip edge endpoints
            if self._vec_equals(p, edge_p1) or self._vec_equals(p, edge_p2):
                continue
            
            # Skip if point is already on a face with this edge
            # This prevents creating duplicate faces
            skip = False
            for face in self.faces:
                if not face.points or len(face.points) < 3:
                    continue
                fp = face.points
                # Check if face contains this edge (in either direction)
                contains_edge = (
                    (self._vec_equals(fp[0], edge_p1) and self._vec_equals(fp[1], edge_p2)) or
                    (self._vec_equals(fp[1], edge_p1) and self._vec_equals(fp[2], edge_p2)) or
                    (self._vec_equals(fp[2], edge_p1) and self._vec_equals(fp[0], edge_p2)) or
                    (self._vec_equals(fp[0], edge_p2) and self._vec_equals(fp[1], edge_p1)) or
                    (self._vec_equals(fp[1], edge_p2) and self._vec_equals(fp[2], edge_p1)) or
                    (self._vec_equals(fp[2], edge_p2) and self._vec_equals(fp[0], edge_p1))
                )
                # Check if point is on this face
                point_on_face = (self._vec_equals(fp[0], p) or 
                                self._vec_equals(fp[1], p) or 
                                self._vec_equals(fp[2], p))
                if contains_edge and point_on_face:
                    skip = True
                    break
                self.comparisons += 6
            
            if skip:
                continue
            
            # Compute perpendicular vector from point to edge
            # This vector lies in the plane perpendicular to the edge
            to_point = p - edge_p1
            proj_on_edge = dot_product(to_point, edge_norm)
            perp_to_edge = to_point - edge_norm * proj_on_edge
            perp_norm = vector_normalize(perp_to_edge, self.epsilon)
            
            # Skip if point is collinear with edge (perpendicular is too small)
            if vector_length(perp_to_edge) < self.epsilon:
                continue
            
            # Compute dihedral angle between current face and candidate face
            if ref_perp is not None and vector_length(ref_perp) > self.epsilon:
                # Use atan2 for robust angle calculation (handles all quadrants)
                cos_angle = dot_product(ref_perp, perp_norm)
                sin_angle = dot_product(edge_norm, cross_product(ref_perp, perp_norm))
                angle = np.arctan2(sin_angle, cos_angle)
                # Normalize angle to [0, 2π)
                if angle < 0:
                    angle += 2 * np.pi
            else:
                # Fallback: use index if we can't compute angle
                angle = i
            
            # Track the point with minimum angle (tightest wrap)
            if angle < min_angle:
                min_angle = angle
                best_idx = i
            self.comparisons += 1
        
        return best_idx
    
    def count_faces_with_edge(self, v1: np.ndarray, v2: np.ndarray) -> int:
        """
        Count how many faces contain a given edge.
        
        Args:
            v1, v2: numpy arrays of shape (3,) representing edge endpoints
            
        Returns:
            int: Number of faces containing this edge
        """
        count = 0
        for face in self.faces:
            if not face.points or len(face.points) < 3:
                continue
            p = face.points
            has_edge = bool(
                (self._vec_equals(p[0], v1) and self._vec_equals(p[1], v2)) or
                (self._vec_equals(p[1], v1) and self._vec_equals(p[2], v2)) or
                (self._vec_equals(p[2], v1) and self._vec_equals(p[0], v2)) or
                (self._vec_equals(p[0], v2) and self._vec_equals(p[1], v1)) or
                (self._vec_equals(p[1], v2) and self._vec_equals(p[2], v1)) or
                (self._vec_equals(p[2], v2) and self._vec_equals(p[0], v1))
            )
            if has_edge:
                count += 1
            self.comparisons += 6
        
        return count
    
    def compute(self, input_points: List[np.ndarray]) -> bool:
        """
        Compute the convex hull using Jarvis March (Gift Wrapping) algorithm.
        
        The algorithm works by:
        1. Finding an initial face (triangle) on the hull
        2. For each edge of the current hull, finding the next point that forms
           the smallest dihedral angle (gift wrapping)
        3. Adding new faces and edges until all edges are processed
        
        Args:
            input_points: List of numpy arrays of shape (3,) representing 3D points
            
        Returns:
            bool: True if hull was computed successfully, False otherwise
        """
        # Need at least 4 points to form a 3D convex hull (tetrahedron)
        if len(input_points) < 4:
            return False
        
        # Initialize algorithm state
        self.vertex_list = [np.array(p) for p in input_points]
        self.faces = []
        self.iterations = 0
        self.comparisons = 0
        
        # Track processed faces to avoid duplicates
        processed_faces: Set[str] = set()
        # Queue of edges to process (edges that need a new face)
        edge_queue: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # Step 1: Find bottommost point (guaranteed to be on hull)
        bottom_idx = self.find_bottommost_point()
        bottom_point = self.vertex_list[bottom_idx]
        
        # Step 2: Find farthest point from bottom (also on hull)
        # This gives us a good initial edge
        max_dist = -1.0
        second_idx = -1
        for i in range(len(self.vertex_list)):
            if i == bottom_idx:
                continue
            # Use squared distance to avoid sqrt (faster)
            v = self.vertex_list[i] - bottom_point
            dist = dot_product(v, v)
            if dist > max_dist:
                max_dist = dist
                second_idx = i
            self.comparisons += 1
        
        if second_idx == -1:
            return False
        
        second_point = self.vertex_list[second_idx]
        
        # Step 3: Find third point maximizing volume
        # This ensures we get a non-degenerate initial triangle
        max_volume = -np.inf
        third_idx = -1
        edge_vec = second_point - bottom_point
        # Find a perpendicular vector to the edge (for reference point)
        edge_perp = cross_product(edge_vec, np.array([0, 0, 1]))
        # If edge is parallel to z-axis, use y-axis instead
        if dot_product(edge_perp, edge_perp) < self.epsilon:
            edge_perp = cross_product(edge_vec, np.array([0, 1, 0]))
        edge_perp = vector_normalize(edge_perp, self.epsilon)
        # Create a reference point slightly offset from the edge
        ref_point = bottom_point + edge_perp * 0.1
        
        # Find point that maximizes volume with bottom and second point
        for i in range(len(self.vertex_list)):
            if i == bottom_idx or i == second_idx:
                continue
            vol = abs(self._signed_volume(bottom_point, second_point, 
                                         self.vertex_list[i], ref_point))
            if vol > max_volume:
                max_volume = vol
                third_idx = i
            self.comparisons += 1
        
        if third_idx == -1:
            return False
        
        third_point = self.vertex_list[third_idx]
        
        # Orientation check: ensure face normal points outward
        # Find a test point not on the initial triangle
        test_point = None
        for i in range(len(self.vertex_list)):
            if i not in [bottom_idx, second_idx, third_idx]:
                test_point = self.vertex_list[i]
                break
        
        # If test point is on positive side, swap second and third to flip normal
        if test_point is not None:
            test_vol = self._signed_volume(bottom_point, second_point, third_point, test_point)
            if test_vol > 0:
                second_point, third_point = third_point, second_point
                second_idx, third_idx = third_idx, second_idx
        
        # Create initial face from the three points
        initial_face = Face(0)
        initial_face.build_from_points(bottom_point, second_point, third_point)
        self.faces.append(initial_face)
        
        # Track this face to avoid duplicates
        face_key = self._face_key(bottom_point, second_point, third_point)
        processed_faces.add(face_key)
        
        # Add all three edges of initial face to the queue
        edge_queue.append((bottom_point, second_point))
        edge_queue.append((second_point, third_point))
        edge_queue.append((third_point, bottom_point))
        
        # Process edges until queue is empty or max iterations reached
        max_iterations = len(self.vertex_list) * len(self.vertex_list)
        skipped_edges = 0
        
        while len(edge_queue) > 0 and self.iterations < max_iterations:
            self.iterations += 1
            # Get next edge from queue (FIFO)
            e1, e2 = edge_queue.pop(0)
            
            # Find the face containing this edge (for normal vector)
            current_face = None
            for face in self.faces:
                if not face.points or len(face.points) < 3:
                    continue
                p = face.points
                # Check if face contains this edge (in either direction)
                contains_edge = bool(
                    (self._vec_equals(p[0], e1) and self._vec_equals(p[1], e2)) or
                    (self._vec_equals(p[1], e1) and self._vec_equals(p[2], e2)) or
                    (self._vec_equals(p[2], e1) and self._vec_equals(p[0], e2)) or
                    (self._vec_equals(p[0], e2) and self._vec_equals(p[1], e1)) or
                    (self._vec_equals(p[1], e2) and self._vec_equals(p[2], e1)) or
                    (self._vec_equals(p[2], e2) and self._vec_equals(p[0], e1))
                )
                if contains_edge:
                    current_face = face
                    break
                self.comparisons += 6
            
            # Skip edges that already have 2 faces (interior edges)
            edge_face_count = self.count_faces_with_edge(e1, e2)
            if edge_face_count >= 2:
                skipped_edges += 1
                continue
            
            # Find the next point to form a face with this edge
            current_normal = current_face.normal if current_face else None
            next_idx = self.find_next_face_point(e1, e2, current_normal)
            
            if next_idx == -1:
                continue
            
            next_point = self.vertex_list[next_idx]
            
            # Compute a reference point inside the hull (for orientation check)
            # Use centroid of all existing face vertices
            inside_ref = None
            if len(self.faces) > 0:
                sum_vec = np.zeros(3)
                cnt = 0
                for f in self.faces:
                    if f.points and len(f.points) >= 3:
                        for pt in f.points:
                            sum_vec += pt
                            cnt += 1
                if cnt > 0:
                    inside_ref = sum_vec / cnt
            
            # Fallback: use midpoint of edge, offset slightly inward
            if inside_ref is None:
                edge_mid = (e1 + e2) / 2.0
                if current_face and current_face.normal is not None:
                    # Offset inward along face normal
                    inside_ref = edge_mid - vector_normalize(current_face.normal, self.epsilon) * 0.1
                else:
                    inside_ref = edge_mid
            
            # Check orientation: ensure new face normal points outward
            face_p1, face_p2, face_p3 = e1, e2, next_point
            test_vol = self._signed_volume(face_p1, face_p2, face_p3, inside_ref)
            # If inside point is on positive side, swap edge endpoints to flip normal
            if test_vol > 0:
                face_p1, face_p2 = face_p2, face_p1
            
            # Check if we've already created this face
            f_key = self._face_key(face_p1, face_p2, face_p3)
            if f_key in processed_faces:
                continue
            
            # Create new face and add to hull
            face = Face(self.iterations)
            face.build_from_points(face_p1, face_p2, face_p3)
            self.faces.append(face)
            processed_faces.add(f_key)
            
            # Add new edges to queue (if they don't already have 2 faces)
            def add_edge_to_queue(v1, v2):
                """Helper to add edge to queue if it needs processing."""
                face_count = self.count_faces_with_edge(v1, v2)
                if face_count < 2:
                    # Check if edge already in queue using vector equality
                    already_in_queue = False
                    for (q1, q2) in edge_queue:
                        if (self._vec_equals(q1, v1) and self._vec_equals(q2, v2)) or \
                           (self._vec_equals(q1, v2) and self._vec_equals(q2, v1)):
                            already_in_queue = True
                            break
                    if not already_in_queue:
                        edge_queue.append((v1, v2))
            
            # Add the two new edges from the new face
            add_edge_to_queue(face_p2, face_p3)
            add_edge_to_queue(face_p3, face_p1)
        
        # Remove duplicate faces (may occur due to numerical precision)
        unique_faces = []
        seen_faces = set()
        for face in self.faces:
            if not face.points or len(face.points) < 3:
                continue
            f_key = self._face_key(face.points[0], face.points[1], face.points[2])
            if f_key not in seen_faces:
                seen_faces.add(f_key)
                unique_faces.append(face)
        
        self.faces = unique_faces
        return True
    
    def _face_key(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> str:
        """
        Create a key for a face (sorted vertex keys).
        
        Args:
            v1, v2, v3: numpy arrays of shape (3,) representing vertices
            
        Returns:
            str: Sorted key string
        """
        keys = [self._vec_key(v1), self._vec_key(v2), self._vec_key(v3)]
        keys.sort()
        return "|".join(keys)
    
    def get_faces(self) -> List[Face]:
        """
        Get all faces in the hull.
        
        Returns:
            List[Face]: List of Face objects
        """
        return self.faces
    
    def get_vertices(self) -> List[np.ndarray]:
        """
        Get all unique vertices in the hull.
        
        Returns:
            List[np.ndarray]: List of unique vertex arrays
        """
        vertices = set()
        for face in self.faces:
            for point in face.points:
                vertices.add(self._vec_key(point))
        return [np.array([float(x) for x in key.split(',')]) for key in vertices]

