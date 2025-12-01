"""
Path Planner Orchestrator

This module coordinates all components of the path planning system:
- Workspace initialization
- Obstacle generation
- Convex hull computation
- A* pathfinding
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from path_planning.workspace import Workspace3D, CellState
from path_planning.obstacles import Obstacle3D, generate_random_obstacles
from path_planning.collision_detection import CollisionDetector
from path_planning.robot import Robot
from path_planning.astar_3d import find_path


class PathPlanner:
    """
    Main path planning orchestrator.
    
    Coordinates workspace, obstacles, convex hulls, and A* pathfinding
    to find a collision-free path from start to goal.
    """
    
    def __init__(self, workspace_size: Tuple[int, int, int] = (20, 20, 20),
                 start_pos: Tuple[int, int, int] = (0, 0, 0),
                 goal_pos: Tuple[int, int, int] = (20, 20, 20)):
        """
        Initialize the path planner.
        
        Args:
            workspace_size: Size of the workspace (size_x, size_y, size_z)
            start_pos: Start position (x, y, z)
            goal_pos: Goal position (x, y, z)
        """
        self.workspace_size = workspace_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        # Initialize workspace
        self.workspace = Workspace3D(workspace_size[0], workspace_size[1], workspace_size[2])
        self.workspace.set_start(start_pos[0], start_pos[1], start_pos[2])
        self.workspace.set_goal(goal_pos[0], goal_pos[1], goal_pos[2])
        
        # Obstacles and collision detection
        self.obstacles: List[Obstacle3D] = []
        self.collision_detector = CollisionDetector()
        
        # Robot
        self.robot = Robot(position=(float(start_pos[0]), float(start_pos[1]), float(start_pos[2])))
        
        # Path
        self.path: Optional[List[Tuple[int, int, int]]] = None
    
    def generate_obstacles(self, num_obstacles: int = 8,
                          obstacle_types: List[str] = None,
                          min_cube_size: float = 2.0,
                          max_cube_size: float = 5.0,
                          min_sphere_radius: float = 1.0,
                          max_sphere_radius: float = 3.0,
                          seed: Optional[int] = None,
                          avoid_start_goal: bool = True) -> List[Obstacle3D]:
        """
        Generate random obstacles in the workspace.
        
        Args:
            num_obstacles: Number of obstacles to generate
            obstacle_types: List of obstacle types ('cube', 'sphere')
            min_cube_size: Minimum cube side length
            max_cube_size: Maximum cube side length
            min_sphere_radius: Minimum sphere radius
            max_sphere_radius: Maximum sphere radius
            seed: Random seed for reproducibility
            avoid_start_goal: If True, ensure obstacles don't block start/goal
        
        Returns:
            List of generated Obstacle3D objects
        """
        workspace_bounds = self.workspace.get_bounds()
        
        # Generate obstacles
        obstacles = generate_random_obstacles(
            num_obstacles=num_obstacles,
            workspace_bounds=workspace_bounds,
            obstacle_types=obstacle_types,
            min_cube_size=min_cube_size,
            max_cube_size=max_cube_size,
            min_sphere_radius=min_sphere_radius,
            max_sphere_radius=max_sphere_radius,
            seed=seed
        )
        
        # Filter obstacles that are too close to start/goal if requested
        if avoid_start_goal:
            filtered_obstacles = []
            start_pos_float = np.array([float(self.start_pos[0]), 
                                       float(self.start_pos[1]), 
                                       float(self.start_pos[2])])
            goal_pos_float = np.array([float(self.goal_pos[0]), 
                                      float(self.goal_pos[1]), 
                                      float(self.goal_pos[2])])
            
            min_distance = 2.0  # Minimum distance from start/goal
            
            for obstacle in obstacles:
                bounds = obstacle.get_bounds()
                # Get center of obstacle
                center = np.array([
                    (bounds[0][0] + bounds[0][1]) / 2.0,
                    (bounds[1][0] + bounds[1][1]) / 2.0,
                    (bounds[2][0] + bounds[2][1]) / 2.0
                ])
                
                dist_to_start = np.linalg.norm(center - start_pos_float)
                dist_to_goal = np.linalg.norm(center - goal_pos_float)
                
                if dist_to_start >= min_distance and dist_to_goal >= min_distance:
                    filtered_obstacles.append(obstacle)
            
            obstacles = filtered_obstacles
        
        self.obstacles = obstacles
        
        # Compute convex hulls for obstacles
        self.collision_detector.compute_obstacle_hulls(obstacles)
        
        # Mark obstacles in workspace grid (approximate)
        self._mark_obstacles_in_workspace()
        
        return obstacles
    
    def _mark_obstacles_in_workspace(self):
        """Mark obstacle regions in the workspace grid (for visualization/debugging)."""
        for obstacle in self.obstacles:
            bounds = obstacle.get_bounds()
            min_x, max_x = int(bounds[0][0]), int(np.ceil(bounds[0][1]))
            min_y, max_y = int(bounds[1][0]), int(np.ceil(bounds[1][1]))
            min_z, max_z = int(bounds[2][0]), int(np.ceil(bounds[2][1]))
            
            # Mark cells in the bounding box as obstacles
            self.workspace.mark_obstacle_region(min_x, max_x, min_y, max_y, min_z, max_z)
    
    def plan_path(self, include_diagonals: bool = True) -> Optional[List[Tuple[int, int, int]]]:
        """
        Plan a path from start to goal using A* algorithm.
        
        Args:
            include_diagonals: If True, use 26-connected neighbors, else 6-connected
        
        Returns:
            List of (x, y, z) tuples representing the path, or None if no path exists
        """
        # Find path using A*
        path = find_path(
            workspace=self.workspace,
            start=self.start_pos,
            goal=self.goal_pos,
            collision_detector=self.collision_detector,
            include_diagonals=include_diagonals
        )
        
        self.path = path
        return path
    
    def get_path_length(self) -> float:
        """
        Compute the total length of the planned path.
        
        Returns:
            float: Total path length, or 0.0 if no path exists
        """
        if self.path is None or len(self.path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(self.path) - 1):
            p1 = np.array(self.path[i], dtype=float)
            p2 = np.array(self.path[i + 1], dtype=float)
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """
        Export workspace, obstacles, hulls, and path data for visualization.
        
        Returns:
            Dictionary containing all data needed for visualization
        """
        # Export workspace bounds
        bounds = self.workspace.get_bounds()
        
        # Export obstacles
        obstacles_data = []
        for i, obstacle in enumerate(self.obstacles):
            points = obstacle.get_obstacle_points()
            points_list = [[float(p[0]), float(p[1]), float(p[2])] for p in points]
            
            bounds = obstacle.get_bounds()  # This now also computes center
            center = obstacle.center if obstacle.center is not None else np.array([
                (bounds[0][0] + bounds[0][1]) / 2.0,
                (bounds[1][0] + bounds[1][1]) / 2.0,
                (bounds[2][0] + bounds[2][1]) / 2.0
            ])
            
            obstacles_data.append({
                'type': obstacle.obstacle_type,
                'points': points_list,
                'bounds': {
                    'x': bounds[0],
                    'y': bounds[1],
                    'z': bounds[2]
                },
                'center': center.tolist()
            })
        
        # Export convex hulls
        hulls_data = []
        for hull in self.collision_detector.get_obstacle_hulls():
            faces = hull.get_faces()
            vertices = hull.get_vertices()
            
            if not vertices or not faces:
                continue
            
            # Export vertices
            vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in vertices]
            
            # Create mapping from vertex to index (with tolerance for floating point comparison)
            def find_vertex_index(point, vertices_array, tolerance=1e-9):
                """Find index of vertex that matches point within tolerance."""
                for idx, vertex in enumerate(vertices_array):
                    if np.allclose(point, vertex, atol=tolerance):
                        return idx
                # Fallback: find closest
                distances = [np.linalg.norm(np.array(v) - np.array(point)) for v in vertices_array]
                return int(np.argmin(distances))
            
            faces_list = []
            for face in faces:
                if len(face.points) >= 3:
                    face_indices = []
                    for point in face.points:
                        idx = find_vertex_index(point, vertices)
                        face_indices.append(idx)
                    # Only add valid triangular faces
                    if len(face_indices) >= 3:
                        faces_list.append(face_indices[:3])  # Use first 3 vertices for triangle
            
            if faces_list:  # Only add hull if it has valid faces
                hulls_data.append({
                    'vertices': vertices_list,
                    'faces': faces_list
                })
        
        # Export path
        path_data = None
        if self.path is not None:
            path_data = [[float(p[0]), float(p[1]), float(p[2])] for p in self.path]
        
        return {
            'workspace': {
                'size': self.workspace_size,
                'bounds': {
                    'x': bounds[0],
                    'y': bounds[1],
                    'z': bounds[2]
                }
            },
            'start': list(self.start_pos),
            'goal': list(self.goal_pos),
            'obstacles': obstacles_data,
            'hulls': hulls_data,
            'path': path_data,
            'robot': {
                'position': self.robot.get_position().tolist(),
                'safety_radius': self.robot.safety_radius
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the path planning problem and solution.
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'workspace_size': self.workspace_size,
            'num_obstacles': len(self.obstacles),
            'path_found': self.path is not None,
            'path_length': self.get_path_length(),
            'path_num_nodes': len(self.path) if self.path else 0
        }
        
        # Count obstacle points
        total_obstacle_points = sum(len(obs.get_obstacle_points()) for obs in self.obstacles)
        stats['total_obstacle_points'] = total_obstacle_points
        
        # Count hull faces
        total_hull_faces = sum(len(hull.get_faces()) for hull in self.collision_detector.get_obstacle_hulls())
        stats['total_hull_faces'] = total_hull_faces
        
        return stats

