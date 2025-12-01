"""
QuickHull 3D Convex Hull Algorithm Implementation.

This module implements the QuickHull algorithm for computing the 3D convex hull
of a set of points. QuickHull is a divide-and-conquer algorithm with average
time complexity O(n log n) and worst-case O(n²).

Time Complexity: O(n log n) average, O(n²) worst case
Space Complexity: O(n)
"""

from typing import List, Optional, Tuple
import numpy as np
from .face import Face, FaceTypes
from .half_edge import HalfEdge
from .geometry_utils import (
    point_to_line_distance,
    signed_distance_to_plane,
    remove_vertex_from_list
)


DISTANCE_TOLERANCE = 0.01


class Quickhull3D:
    """
    QuickHull 3D convex hull algorithm implementation.
    
    This class implements the QuickHull algorithm which builds the convex hull
    by starting with an initial tetrahedron and iteratively adding points that
    are outside the current hull.
    """
    
    def __init__(self):
        """
        Initialize a QuickHull3D instance.
        
        Sets up data structures for the QuickHull algorithm:
        - vertex_list: Input points to compute hull for
        - claimed: Points that are outside some face (candidates for expansion)
        - vertex_to_face: Mapping from vertices to their associated faces
        - unclaimed: Points temporarily unclaimed during face deletion
        - horizon: Boundary edges between visible and non-visible faces
        - new_faces: Faces created in the current iteration
        - faces: All faces in the current hull (including deleted ones)
        - total_steps: Total number of algorithm steps
        - iterations: Counter for algorithm iterations (for benchmarking)
        - comparisons: Counter for comparison operations (for benchmarking)
        """
        self.vertex_list: List[np.ndarray] = []  # List of input points (numpy arrays)
        self.claimed: List[np.ndarray] = []  # Points claimed by faces (outside set)
        self.vertex_to_face: dict = {}  # Map vertex to face (replaces v.face attribute)
        self.unclaimed: List[np.ndarray] = []  # Unclaimed points (temporary during deletion)
        self.horizon: List[HalfEdge] = []  # Horizon edges (boundary of visible region)
        self.new_faces: List[Face] = []  # Newly created faces in current iteration
        self.faces: List[Face] = []  # All faces in the hull (including deleted)
        self.total_steps: int = 0  # Total number of steps
        self.iterations: int = 0  # Operation counter for benchmarking
        self.comparisons: int = 0  # Comparison counter for benchmarking
    
    def build_initial_simplex(self, step: int):
        """
        Build the initial tetrahedron from the input points.
        
        The initial simplex (tetrahedron) is the starting point for QuickHull.
        We select 4 points that form a well-distributed tetrahedron:
        1. Two points with maximum 1D distance (farthest apart in one dimension)
        2. Third point with maximum distance to the line through first two
        3. Fourth point with maximum distance to the plane through first three
        
        This ensures we start with a non-degenerate tetrahedron that contains
        or is close to containing all other points.
        
        Args:
            step: Current step number for face creation tracking
        """
        # Find min and max vertices in each dimension (x, y, z)
        # This helps us find the dimension with maximum spread
        min_vertices = [
            np.array([np.inf, np.inf, np.inf]),
            np.array([np.inf, np.inf, np.inf]),
            np.array([np.inf, np.inf, np.inf])
        ]
        max_vertices = [
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([-np.inf, -np.inf, -np.inf])
        ]
        
        # Track extreme points in each dimension
        for v in self.vertex_list:
            for i in range(3):
                if v[i] < min_vertices[i][i]:
                    min_vertices[i] = v.copy()
                if v[i] > max_vertices[i][i]:
                    max_vertices[i] = v.copy()
                self.comparisons += 2
        
        # Find dimension with maximum spread (x, y, or z)
        # This gives us the best initial edge
        max_distance = max_vertices[0][0] - min_vertices[0][0]
        max_axis = 0
        for i in range(1, 3):
            distance = max_vertices[i][i] - min_vertices[i][i]
            if distance > max_distance:
                max_distance = distance
                max_axis = i
            self.comparisons += 1
        
        # First two vertices: min and max in the dimension with max spread
        v1 = min_vertices[max_axis]
        v2 = max_vertices[max_axis]
        
        # Find third point: maximum distance to line v1-v2
        # This ensures the triangle has good area
        max_dist = -np.inf
        max_vertex = None
        for v in self.vertex_list:
            distance = point_to_line_distance(v2, v1, v)
            if distance > max_dist:
                max_dist = distance
                max_vertex = v
            self.comparisons += 1
        
        v3 = max_vertex
        
        # Find fourth point: maximum distance to plane v1-v2-v3
        # This ensures the tetrahedron has good volume
        max_dist = -np.inf
        max_vertex = None
        for v in self.vertex_list:
            distance = abs(signed_distance_to_plane(v1, v2, v3, v))
            if distance > max_dist:
                max_dist = distance
                max_vertex = v
            self.comparisons += 1
        
        v4 = max_vertex
        
        # Determine orientation: check if v4 is in front of or behind the plane v1-v2-v3
        # This determines how we order the faces
        signed_dist = signed_distance_to_plane(v1, v2, v3, v4)
        v4_in_front = signed_dist > 0
        
        # Create 4 faces of the tetrahedron
        # Face ordering depends on whether v4 is in front or behind the base triangle
        faces = [Face(step) for _ in range(4)]
        
        if v4_in_front:
            # v4 is in front: order faces so normals point outward
            faces[0].build_from_points(v3, v2, v1)  # Base face (opposite v4)
            faces[1].build_from_points(v2, v3, v4)  # Face containing v2, v3, v4
            faces[2].build_from_points(v4, v3, v1)  # Face containing v4, v3, v1
            faces[3].build_from_points(v1, v2, v4)  # Face containing v1, v2, v4
        else:
            # v4 is behind: reverse face ordering
            faces[0].build_from_points(v1, v2, v3)  # Base face (opposite v4)
            faces[1].build_from_points(v4, v3, v2)  # Face containing v4, v3, v2
            faces[2].build_from_points(v1, v3, v4)  # Face containing v1, v3, v4
            faces[3].build_from_points(v4, v2, v1)  # Face containing v4, v2, v1
        
        # Connect faces via twin edges (establish half-edge connectivity)
        # Each edge is shared by exactly two faces, so we link their half-edges
        faces[0].find_edge_with_extremities(v1, v2).set_twin(
            faces[3].find_edge_with_extremities(v1, v2)
        )
        faces[0].find_edge_with_extremities(v2, v3).set_twin(
            faces[1].find_edge_with_extremities(v2, v3)
        )
        faces[0].find_edge_with_extremities(v1, v3).set_twin(
            faces[2].find_edge_with_extremities(v1, v3)
        )
        
        faces[1].find_edge_with_extremities(v3, v4).set_twin(
            faces[2].find_edge_with_extremities(v3, v4)
        )
        faces[1].find_edge_with_extremities(v2, v4).set_twin(
            faces[3].find_edge_with_extremities(v2, v4)
        )
        
        faces[2].find_edge_with_extremities(v1, v4).set_twin(
            faces[3].find_edge_with_extremities(v1, v4)
        )
        
        self.faces.extend(faces)
        
        # Assign remaining points to faces (outside set)
        # Each point is assigned to the face it's farthest from (if outside)
        for v in self.vertex_list:
            # Skip the four vertices of the tetrahedron
            if (np.allclose(v, v1) or np.allclose(v, v2) or 
                np.allclose(v, v3) or np.allclose(v, v4)):
                continue
            
            # Find the face this point is farthest from (if outside)
            max_dist = DISTANCE_TOLERANCE
            max_dist_face = None
            for face in faces:
                dist = face.signed_distance_from_point(v)
                if dist > max_dist:
                    max_dist = dist
                    max_dist_face = face
                self.comparisons += 1
            
            # Add point to the outside set of the farthest face
            if max_dist_face is not None:
                self.add_point_to_face(v, max_dist_face)
    
    def add_point_to_face(self, v: np.ndarray, f: Face):
        """
        Add a point to the outside set of a face.
        
        Args:
            v: numpy array of shape (3,) representing the vertex
            f: Face object to associate the vertex with
        """
        # Store vertex-to-face mapping (replaces v.face attribute)
        self.vertex_to_face[id(v)] = f
        self.claimed.append(v)
        f.outside.append(v)
    
    def next_point_to_add(self) -> Optional[np.ndarray]:
        """
        Find the next point to add to the hull.
        
        Selects the point with maximum distance from its associated face.
        This point (called the "eye") is guaranteed to be outside the current hull
        and will expand the hull when added. The point with maximum distance is
        chosen to maximize progress and ensure numerical stability.
        
        Returns:
            Optional numpy array: The next point to add (eye vertex), or None if no points remain
        """
        if len(self.claimed) == 0:
            return None
        
        # Get the face associated with the first claimed point
        eye_face = self.vertex_to_face.get(id(self.claimed[0]))
        if eye_face is None:
            return None
        
        # Find the point in this face's outside set that is farthest from the face
        eye_vertex = None
        max_dist = 0.0
        
        for vertex in eye_face.outside:
            dist = eye_face.signed_distance_from_point(vertex)
            if dist > max_dist:
                max_dist = dist
                eye_vertex = vertex
            self.comparisons += 1
        
        return eye_vertex
    
    def add_point_to_hull(self, eye: np.ndarray, step: int):
        """
        Add a point to the hull by removing visible faces and creating new ones.
        
        This is the core operation of QuickHull. When adding a point:
        1. Find all faces visible from the point (faces the point is outside of)
        2. Calculate the horizon (boundary between visible and non-visible faces)
        3. Remove visible faces
        4. Create new faces connecting the point to horizon edges
        5. Reassign unclaimed points to new faces
        
        Args:
            eye: numpy array of shape (3,) representing the point to add
            step: Current step number for tracking face creation
        """
        # Initialize data structures for this iteration
        self.horizon = []
        self.unclaimed = []
        
        # Get the face that claimed this point
        eye_face = self.vertex_to_face.get(id(eye))
        if eye_face is None:
            return
        
        # Remove the eye point from its face's outside set
        self.remove_point_from_face(eye, eye_face)
        # Calculate the horizon (boundary of visible region)
        self.calculate_horizon(eye, None, eye_face, self.horizon, step)
        # Create new faces connecting eye to horizon edges
        self.new_faces = self.add_new_faces(eye, self.horizon, step)
        # Reassign points from deleted faces to new faces
        self.resolve_unclaimed_points(self.new_faces)
        self.iterations += 1
    
    def resolve_unclaimed_points(self, new_faces: List[Face]):
        """
        Reassign unclaimed points to new faces.
        
        Args:
            new_faces: List of newly created Face objects
        """
        for unclaimed_vert in self.unclaimed:
            max_dist = DISTANCE_TOLERANCE
            max_face = None
            
            for f in new_faces:
                if f.mark == FaceTypes.VISIBLE:
                    dist = f.signed_distance_from_point(unclaimed_vert)
                    if dist > max_dist:
                        max_dist = dist
                        max_face = f
                    self.comparisons += 1
            
            if max_face is not None:
                self.add_point_to_face(unclaimed_vert, max_face)
    
    def add_new_faces(self, eye: np.ndarray, horizon: List[HalfEdge], step: int) -> List[Face]:
        """
        Create new faces connecting the eye point to horizon edges.
        
        Args:
            eye: numpy array of shape (3,) representing the point to connect
            horizon: List of HalfEdge objects forming the horizon
            step: Current step number
            
        Returns:
            List[Face]: Newly created faces
        """
        new_faces = []
        edge_prev = None
        edge_begin = None
        
        for hedge in horizon:
            hedge_side = self.add_adjoining_face(eye, hedge, step)
            if edge_prev is not None:
                edge_prev.prev.set_twin(hedge_side.next)
            else:
                edge_begin = hedge_side
            
            new_faces.append(hedge_side.face)
            edge_prev = hedge_side
        
        if edge_begin is not None and edge_prev is not None:
            edge_begin.prev.set_twin(edge_prev.next)
        
        return new_faces
    
    def add_adjoining_face(self, eye: np.ndarray, edge: HalfEdge, step: int) -> HalfEdge:
        """
        Create a new face adjoining an edge.
        
        Args:
            eye: numpy array of shape (3,) representing the new vertex
            edge: HalfEdge to connect to
            step: Current step number
            
        Returns:
            HalfEdge: The new half-edge of the created face
        """
        face = Face(step)
        face.build_from_point_and_half_edge(eye, edge)
        self.faces.append(face)
        face.half_edges[2].set_twin(edge.twin)
        return face.half_edges[2]
    
    def remove_point_from_face(self, v: np.ndarray, f: Face):
        """
        Remove a point from a face's outside set.
        
        Args:
            v: numpy array of shape (3,) representing the vertex
            f: Face object to remove the vertex from
        """
        f.remove_vertex_from_outside_set(v)
        remove_vertex_from_list(v, self.claimed)
        if id(v) in self.vertex_to_face:
            del self.vertex_to_face[id(v)]
    
    def remove_all_points_from_face(self, f: Face) -> List[np.ndarray]:
        """
        Remove all points from a face's outside set.
        
        Args:
            f: Face object
            
        Returns:
            List[np.ndarray]: List of removed vertices
        """
        removed_pts = f.outside.copy()
        for v in removed_pts:
            self.remove_point_from_face(v, f)
        return removed_pts
    
    def calculate_horizon(self, eye: np.ndarray, edge0: Optional[HalfEdge], 
                         face: Face, horizon: List[HalfEdge], step: int):
        """
        Recursively calculate the horizon edges.
        
        The horizon is the boundary between visible and non-visible faces.
        It consists of edges where one adjacent face is visible and the other is not.
        This function recursively traverses visible faces, marking them as deleted
        and collecting horizon edges.
        
        Args:
            eye: numpy array of shape (3,) representing the point being added
            edge0: Optional starting edge (None for first call, used for recursion)
            face: Current face being processed (must be visible)
            horizon: List to accumulate horizon edges
            step: Current step number for marking deleted faces
        """
        # Remove points from this face and mark it as deleted
        self.delete_face_points(face, None)
        face.mark_as_deleted(step)
        
        # Start from the given edge, or the first edge of the face
        if edge0 is None:
            edge0 = face.half_edges[0]
            edge = edge0
        else:
            edge = edge0.next
        
        # Traverse all edges of this face
        while True:
            # Get the face on the opposite side of this edge
            opp_face = edge.opposite_face()
            if opp_face is not None and opp_face.mark == FaceTypes.VISIBLE:
                # Check if the opposite face is also visible
                if opp_face.signed_distance_from_point(eye) > DISTANCE_TOLERANCE:
                    # Opposite face is visible: recurse into it
                    self.calculate_horizon(eye, edge.twin, opp_face, horizon, step)
                else:
                    # Opposite face is not visible: this edge is part of the horizon
                    horizon.append(edge)
                self.comparisons += 1
            
            # Move to next edge
            edge = edge.next
            # Stop when we've completed a full cycle
            if edge == edge0:
                break
    
    def delete_face_points(self, face: Face, absorbing_face: Optional[Face]):
        """
        Remove points from a face and reassign them.
        
        Args:
            face: Face object to remove points from
            absorbing_face: Optional face to absorb the points
        """
        face_verts = self.remove_all_points_from_face(face)
        if face_verts:
            if absorbing_face is None:
                self.unclaimed.extend(face_verts)
            else:
                for v in face_verts:
                    dist = absorbing_face.signed_distance_from_point(v)
                    if dist > DISTANCE_TOLERANCE:
                        self.add_point_to_face(v, absorbing_face)
                    else:
                        self.unclaimed.append(v)
                    self.comparisons += 1
    
    def build(self, input_points: List[np.ndarray]) -> bool:
        """
        Build the convex hull from input points using QuickHull algorithm.
        
        The algorithm works by:
        1. Building an initial tetrahedron from 4 well-distributed points
        2. Iteratively adding points that are outside the current hull
        3. For each point, removing visible faces and creating new ones
        4. Continuing until no more points are outside the hull
        
        Args:
            input_points: List of numpy arrays of shape (3,) representing 3D points
            
        Returns:
            bool: True if hull was built successfully, False otherwise
        """
        # Need at least 4 points to form a 3D convex hull (tetrahedron)
        if len(input_points) < 4:
            return False
        
        # Initialize algorithm state
        step = 0
        self.vertex_list = [np.array(p) for p in input_points]
        self.claimed = []
        self.vertex_to_face = {}
        self.unclaimed = []
        self.faces = []
        self.iterations = 0
        self.comparisons = 0
        
        # Step 1: Build initial tetrahedron and assign points to faces
        self.build_initial_simplex(step)
        step += 1
        
        # Step 2: Iteratively add points until no more points are outside
        eye = self.next_point_to_add()
        while eye is not None:
            # Add the point, removing visible faces and creating new ones
            self.add_point_to_hull(eye, step)
            step += 1
            # Find the next point to add
            eye = self.next_point_to_add()
            self.iterations += 1
        
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
                # Use tuple for hashing
                vertices.add(tuple(point))
        return [np.array(v) for v in vertices]

