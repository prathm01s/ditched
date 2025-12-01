"""
3D A* Pathfinding Algorithm

This module implements the A* pathfinding algorithm for 3D grid-based navigation
with collision detection using convex hulls.
"""

from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import heapq
from path_planning.workspace import Workspace3D
from path_planning.collision_detection import CollisionDetector


class Node:
    """
    Node in the A* search graph.
    """
    
    def __init__(self, x: int, y: int, z: int, g_cost: float = 0.0, 
                 h_cost: float = 0.0, parent: Optional['Node'] = None):
        """
        Initialize a node.
        
        Args:
            x, y, z: Grid coordinates
            g_cost: Cost from start to this node
            h_cost: Heuristic cost from this node to goal
            parent: Parent node in the path
        """
        self.x = x
        self.y = y
        self.z = z
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other):
        """Comparison for priority queue (heap)."""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return self.h_cost < other.h_cost
    
    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self):
        """Hash for use in sets/dictionaries."""
        return hash((self.x, self.y, self.z))
    
    def get_position(self) -> Tuple[int, int, int]:
        """Get the position as a tuple."""
        return (self.x, self.y, self.z)
    
    def get_position_float(self) -> np.ndarray:
        """Get the position as a float numpy array (for collision detection)."""
        return np.array([float(self.x), float(self.y), float(self.z)])


def heuristic_3d(pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
    """
    Compute 3D Euclidean distance heuristic.
    
    Args:
        pos1: First position (x, y, z)
        pos2: Second position (x, y, z)
    
    Returns:
        float: Euclidean distance between positions
    """
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def get_neighbors_3d(x: int, y: int, z: int, workspace: Workspace3D,
                    include_diagonals: bool = True) -> List[Tuple[int, int, int]]:
    """
    Get all valid neighboring cells in 3D space.
    
    Args:
        x, y, z: Current position
        workspace: Workspace3D instance
        include_diagonals: If True, include diagonal neighbors (26-connected),
                          otherwise only face neighbors (6-connected)
    
    Returns:
        List of (x, y, z) tuples representing valid neighbor positions
    """
    return workspace.get_neighbors(x, y, z, include_diagonals)


def get_edge_cost(pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
    """
    Compute the cost of moving from pos1 to pos2.
    
    Args:
        pos1: Start position (x, y, z)
        pos2: End position (x, y, z)
    
    Returns:
        float: Cost of the edge (distance)
    """
    return heuristic_3d(pos1, pos2)


def reconstruct_path(node: Node) -> List[Tuple[int, int, int]]:
    """
    Reconstruct the path from start to goal by following parent pointers.
    
    Args:
        node: Goal node
    
    Returns:
        List of (x, y, z) tuples representing the path from start to goal
    """
    path = []
    current = node
    
    while current is not None:
        path.append(current.get_position())
        current = current.parent
    
    path.reverse()
    return path


def find_path(workspace: Workspace3D,
              start: Tuple[int, int, int],
              goal: Tuple[int, int, int],
              collision_detector: Optional[CollisionDetector] = None,
              include_diagonals: bool = True) -> Optional[List[Tuple[int, int, int]]]:
    """
    Find a path from start to goal using A* algorithm.
    
    Args:
        workspace: Workspace3D instance
        start: Start position (x, y, z)
        goal: Goal position (x, y, z)
        collision_detector: Optional CollisionDetector for collision checking
        include_diagonals: If True, use 26-connected neighbors, else 6-connected
    
    Returns:
        List of (x, y, z) tuples representing the path, or None if no path exists
    """
    # Validate start and goal positions
    if not workspace.is_valid_position(*start) or not workspace.is_valid_position(*goal):
        return None
    
    # Check if start or goal are obstacles (using collision detector if provided)
    if collision_detector is not None:
        start_pos_float = np.array([float(start[0]), float(start[1]), float(start[2])])
        goal_pos_float = np.array([float(goal[0]), float(goal[1]), float(goal[2])])
        
        if collision_detector.check_collision(start_pos_float):
            return None
        if collision_detector.check_collision(goal_pos_float):
            return None
    
    # Initialize open and closed sets
    open_set: List[Node] = []
    closed_set: Set[Tuple[int, int, int]] = set()
    
    # Create start node
    start_node = Node(start[0], start[1], start[2], 
                     g_cost=0.0, 
                     h_cost=heuristic_3d(start, goal))
    
    heapq.heappush(open_set, start_node)
    
    # Dictionary to track best g_cost for each position
    g_costs: Dict[Tuple[int, int, int], float] = {start: 0.0}
    
    while open_set:
        # Get node with lowest f_cost
        current = heapq.heappop(open_set)
        current_pos = current.get_position()
        
        # Skip if already processed
        if current_pos in closed_set:
            continue
        
        # Add to closed set
        closed_set.add(current_pos)
        
        # Check if we reached the goal
        if current_pos == goal:
            return reconstruct_path(current)
        
        # Explore neighbors
        neighbors = get_neighbors_3d(current.x, current.y, current.z, workspace, include_diagonals)
        
        for neighbor_pos in neighbors:
            nx, ny, nz = neighbor_pos
            
            # Skip if already in closed set
            if neighbor_pos in closed_set:
                continue
            
            # Check collision if collision detector is provided
            if collision_detector is not None:
                neighbor_pos_float = np.array([float(nx), float(ny), float(nz)])
                
                # Check if neighbor position collides
                if collision_detector.check_collision(neighbor_pos_float):
                    continue
                
                # Check if path segment from current to neighbor is safe
                current_pos_float = current.get_position_float()
                if not collision_detector.is_path_safe(current_pos_float, neighbor_pos_float):
                    continue
            
            # Compute edge cost
            edge_cost = get_edge_cost(current_pos, neighbor_pos)
            tentative_g_cost = current.g_cost + edge_cost
            
            # Check if we found a better path to this neighbor
            if neighbor_pos not in g_costs or tentative_g_cost < g_costs[neighbor_pos]:
                g_costs[neighbor_pos] = tentative_g_cost
                
                # Compute heuristic
                h_cost = heuristic_3d(neighbor_pos, goal)
                
                # Create neighbor node
                neighbor_node = Node(nx, ny, nz,
                                   g_cost=tentative_g_cost,
                                   h_cost=h_cost,
                                   parent=current)
                
                heapq.heappush(open_set, neighbor_node)
    
    # No path found
    return None

