"""
3D Workspace Grid Representation

This module provides a 3D grid-based workspace for robot path planning.
The workspace is a 20x20x20 grid where each cell can be FREE, OBSTACLE, START, or GOAL.
"""

from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class CellState(Enum):
    """Enumeration of possible cell states in the grid."""
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3


class Workspace3D:
    """
    3D grid-based workspace for robot path planning.
    
    The workspace is a cubic grid from (0,0,0) to (size_x, size_y, size_z).
    Each cell can be in one of four states: FREE, OBSTACLE, START, or GOAL.
    """
    
    def __init__(self, size_x: int = 20, size_y: int = 20, size_z: int = 20):
        """
        Initialize a 3D workspace grid.
        
        Args:
            size_x: Size of the workspace in x-direction (default: 20)
            size_y: Size of the workspace in y-direction (default: 20)
            size_z: Size of the workspace in z-direction (default: 20)
        """
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        
        # Initialize grid with all cells set to FREE
        self.grid = np.full((size_x, size_y, size_z), CellState.FREE, dtype=object)
        
        self.start_pos: Optional[Tuple[int, int, int]] = None
        self.goal_pos: Optional[Tuple[int, int, int]] = None
    
    def is_valid_position(self, x: int, y: int, z: int) -> bool:
        """
        Check if a position is within the workspace bounds.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if position is within bounds, False otherwise
        """
        return (0 <= x < self.size_x and 
                0 <= y < self.size_y and 
                0 <= z < self.size_z)
    
    def get_cell_state(self, x: int, y: int, z: int) -> Optional[CellState]:
        """
        Get the state of a cell at the given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            CellState or None: The state of the cell, or None if position is invalid
        """
        if not self.is_valid_position(x, y, z):
            return None
        return self.grid[x, y, z]
    
    def set_cell_state(self, x: int, y: int, z: int, state: CellState) -> bool:
        """
        Set the state of a cell at the given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            state: The new state for the cell
            
        Returns:
            bool: True if the cell was set successfully, False if position is invalid
        """
        if not self.is_valid_position(x, y, z):
            return False
        self.grid[x, y, z] = state
        return True
    
    def mark_obstacle(self, x: int, y: int, z: int) -> bool:
        """
        Mark a cell as an obstacle.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if the cell was marked successfully
        """
        return self.set_cell_state(x, y, z, CellState.OBSTACLE)
    
    def mark_obstacle_region(self, min_x: int, max_x: int, 
                            min_y: int, max_y: int,
                            min_z: int, max_z: int) -> int:
        """
        Mark a rectangular region as obstacles.
        
        Args:
            min_x, max_x: X bounds (inclusive)
            min_y, max_y: Y bounds (inclusive)
            min_z, max_z: Z bounds (inclusive)
            
        Returns:
            int: Number of cells marked as obstacles
        """
        count = 0
        for x in range(max(0, min_x), min(self.size_x, max_x + 1)):
            for y in range(max(0, min_y), min(self.size_y, max_y + 1)):
                for z in range(max(0, min_z), min(self.size_z, max_z + 1)):
                    if self.mark_obstacle(x, y, z):
                        count += 1
        return count
    
    def set_start(self, x: int, y: int, z: int) -> bool:
        """
        Set the start position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if the start position was set successfully
        """
        if not self.is_valid_position(x, y, z):
            return False
        
        # Clear previous start position
        if self.start_pos is not None:
            sx, sy, sz = self.start_pos
            if self.grid[sx, sy, sz] == CellState.START:
                self.grid[sx, sy, sz] = CellState.FREE
        
        self.start_pos = (x, y, z)
        self.grid[x, y, z] = CellState.START
        return True
    
    def set_goal(self, x: int, y: int, z: int) -> bool:
        """
        Set the goal position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if the goal position was set successfully
        """
        if not self.is_valid_position(x, y, z):
            return False
        
        # Clear previous goal position
        if self.goal_pos is not None:
            gx, gy, gz = self.goal_pos
            if self.grid[gx, gy, gz] == CellState.GOAL:
                self.grid[gx, gy, gz] = CellState.FREE
        
        self.goal_pos = (x, y, z)
        self.grid[x, y, z] = CellState.GOAL
        return True
    
    def is_free(self, x: int, y: int, z: int) -> bool:
        """
        Check if a cell is free (not an obstacle).
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if the cell is free, False otherwise
        """
        if not self.is_valid_position(x, y, z):
            return False
        state = self.grid[x, y, z]
        return state == CellState.FREE or state == CellState.START or state == CellState.GOAL
    
    def get_neighbors(self, x: int, y: int, z: int, 
                     include_diagonals: bool = True) -> List[Tuple[int, int, int]]:
        """
        Get all valid neighboring cells.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            include_diagonals: If True, include diagonal neighbors (26-connected),
                              otherwise only face neighbors (6-connected)
            
        Returns:
            List of (x, y, z) tuples representing valid neighbor positions
        """
        neighbors = []
        
        if include_diagonals:
            # 26-connected neighbors (all adjacent cells including diagonals)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if self.is_valid_position(nx, ny, nz):
                            neighbors.append((nx, ny, nz))
        else:
            # 6-connected neighbors (only face-adjacent)
            for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), 
                               (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if self.is_valid_position(nx, ny, nz):
                    neighbors.append((nx, ny, nz))
        
        return neighbors
    
    def clear(self):
        """Clear all obstacles and reset start/goal positions."""
        self.grid.fill(CellState.FREE)
        self.start_pos = None
        self.goal_pos = None
    
    def get_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the bounds of the workspace.
        
        Returns:
            Tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        """
        return ((0, self.size_x - 1), (0, self.size_y - 1), (0, self.size_z - 1))

