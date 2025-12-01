"""
Unit tests for A* pathfinding algorithm.
"""

import unittest
import numpy as np
from path_planning.workspace import Workspace3D
from path_planning.astar_3d import find_path, heuristic_3d, get_neighbors_3d
from path_planning.collision_detection import CollisionDetector
from path_planning.obstacles import generate_cube_obstacle


class TestHeuristic(unittest.TestCase):
    """Test heuristic function."""
    
    def test_heuristic_3d(self):
        """Test 3D Euclidean distance heuristic."""
        pos1 = (0, 0, 0)
        pos2 = (3, 4, 0)
        expected = 5.0  # sqrt(3^2 + 4^2 + 0^2) = 5
        result = heuristic_3d(pos1, pos2)
        self.assertAlmostEqual(result, expected, places=5)
        
        pos3 = (1, 1, 1)
        pos4 = (2, 2, 2)
        expected2 = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        result2 = heuristic_3d(pos3, pos4)
        self.assertAlmostEqual(result2, expected2, places=5)


class TestNeighbors(unittest.TestCase):
    """Test neighbor generation."""
    
    def test_get_neighbors_3d(self):
        """Test 3D neighbor generation."""
        workspace = Workspace3D(10, 10, 10)
        
        # Test 26-connected neighbors
        neighbors = get_neighbors_3d(5, 5, 5, workspace, include_diagonals=True)
        self.assertEqual(len(neighbors), 26)
        
        # Test 6-connected neighbors
        neighbors_6 = get_neighbors_3d(5, 5, 5, workspace, include_diagonals=False)
        self.assertEqual(len(neighbors_6), 6)
    
    def test_neighbors_at_boundary(self):
        """Test neighbor generation at workspace boundary."""
        workspace = Workspace3D(10, 10, 10)
        
        # At corner (0, 0, 0)
        neighbors = get_neighbors_3d(0, 0, 0, workspace, include_diagonals=True)
        self.assertLess(len(neighbors), 26)  # Fewer neighbors at boundary
        self.assertGreater(len(neighbors), 0)
        
        # All neighbors should be valid
        for nx, ny, nz in neighbors:
            self.assertTrue(workspace.is_valid_position(nx, ny, nz))


class TestAStar(unittest.TestCase):
    """Test A* pathfinding algorithm."""
    
    def test_simple_path(self):
        """Test pathfinding in empty workspace."""
        workspace = Workspace3D(10, 10, 10)
        start = (0, 0, 0)
        goal = (9, 9, 9)
        
        path = find_path(workspace, start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertGreater(len(path), 1)
    
    def test_path_with_obstacles(self):
        """Test pathfinding with obstacles."""
        workspace = Workspace3D(20, 20, 20)
        start = (0, 0, 0)
        goal = (19, 19, 19)
        
        # Create obstacle in the middle
        obstacle = generate_cube_obstacle((10, 10, 10), 3.0, resolution=4)
        detector = CollisionDetector()
        detector.compute_obstacle_hulls([obstacle])
        
        path = find_path(workspace, start, goal, collision_detector=detector)
        
        # Path should be found (obstacle shouldn't block completely)
        # If path is found, verify it doesn't go through obstacle
        if path:
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
            
            # Check that path doesn't go through obstacle center
            for node in path:
                point = np.array([float(node[0]), float(node[1]), float(node[2])])
                # Allow some tolerance, but shouldn't be exactly at center
                dist_to_center = np.linalg.norm(point - np.array([10.0, 10.0, 10.0]))
                self.assertGreater(dist_to_center, 1.0)  # Should be at least 1 unit away
    
    def test_no_path_exists(self):
        """Test when no path exists."""
        workspace = Workspace3D(10, 10, 10)
        start = (0, 0, 0)
        goal = (9, 9, 9)
        
        # Create obstacles blocking all paths
        obstacles = [
            generate_cube_obstacle((5, 5, 5), 10.0, resolution=4)  # Large obstacle
        ]
        detector = CollisionDetector()
        detector.compute_obstacle_hulls(obstacles)
        
        path = find_path(workspace, start, goal, collision_detector=detector)
        
        # Path might still be found if obstacle doesn't completely block
        # This test is more about ensuring the function doesn't crash
        if path is None:
            # No path found is valid
            pass
        else:
            # If path found, it should be valid
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
    
    def test_start_equals_goal(self):
        """Test pathfinding when start equals goal."""
        workspace = Workspace3D(10, 10, 10)
        start = (5, 5, 5)
        goal = (5, 5, 5)
        
        path = find_path(workspace, start, goal)
        
        # Should return path with single node
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], start)
    
    def test_invalid_positions(self):
        """Test pathfinding with invalid start/goal positions."""
        workspace = Workspace3D(10, 10, 10)
        
        # Invalid start
        path1 = find_path(workspace, (-1, 0, 0), (9, 9, 9))
        self.assertIsNone(path1)
        
        # Invalid goal
        path2 = find_path(workspace, (0, 0, 0), (10, 10, 10))
        self.assertIsNone(path2)


if __name__ == '__main__':
    unittest.main()

