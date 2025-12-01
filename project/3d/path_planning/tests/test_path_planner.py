"""
Integration tests for path planner.
"""

import unittest
import numpy as np
from path_planning.path_planner import PathPlanner
from path_planning.obstacles import generate_cube_obstacle, generate_sphere_obstacle


class TestPathPlanner(unittest.TestCase):
    """Test PathPlanner class."""
    
    def test_basic_path_planning(self):
        """Test basic path planning without obstacles."""
        planner = PathPlanner(
            workspace_size=(10, 10, 10),
            start_pos=(0, 0, 0),
            goal_pos=(9, 9, 9)
        )
        
        path = planner.plan_path()
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0, 0))
        self.assertEqual(path[-1], (9, 9, 9))
        self.assertGreater(len(path), 1)
    
    def test_path_planning_with_obstacles(self):
        """Test path planning with obstacles."""
        planner = PathPlanner(
            workspace_size=(20, 20, 20),
            start_pos=(0, 0, 0),
            goal_pos=(19, 19, 19)
        )
        
        # Generate obstacles
        obstacles = planner.generate_obstacles(
            num_obstacles=5,
            obstacle_types=['cube'],
            seed=42
        )
        
        self.assertGreater(len(obstacles), 0)
        self.assertGreater(len(planner.collision_detector.get_obstacle_hulls()), 0)
        
        # Plan path
        path = planner.plan_path()
        
        # Path should be found (or None if completely blocked)
        if path:
            self.assertEqual(path[0], (0, 0, 0))
            self.assertEqual(path[-1], (19, 19, 19))
            self.assertGreater(len(path), 1)
    
    def test_path_statistics(self):
        """Test path statistics computation."""
        planner = PathPlanner(
            workspace_size=(10, 10, 10),
            start_pos=(0, 0, 0),
            goal_pos=(9, 9, 9)
        )
        
        path = planner.plan_path()
        
        if path:
            stats = planner.get_statistics()
            
            self.assertTrue(stats['path_found'])
            self.assertGreater(stats['path_length'], 0)
            self.assertGreater(stats['path_num_nodes'], 0)
            self.assertEqual(stats['workspace_size'], (10, 10, 10))
    
    def test_export_for_visualization(self):
        """Test data export for visualization."""
        planner = PathPlanner(
            workspace_size=(10, 10, 10),
            start_pos=(0, 0, 0),
            goal_pos=(9, 9, 9)
        )
        
        planner.generate_obstacles(num_obstacles=3, seed=42)
        path = planner.plan_path()
        
        if path:
            data = planner.export_for_visualization()
            
            self.assertIn('workspace', data)
            self.assertIn('start', data)
            self.assertIn('goal', data)
            self.assertIn('obstacles', data)
            self.assertIn('hulls', data)
            self.assertIn('path', data)
            self.assertIn('robot', data)
            
            self.assertEqual(data['start'], [0, 0, 0])
            self.assertEqual(data['goal'], [9, 9, 9])
            self.assertIsNotNone(data['path'])
    
    def test_obstacle_generation(self):
        """Test obstacle generation."""
        planner = PathPlanner(
            workspace_size=(20, 20, 20),
            start_pos=(0, 0, 0),
            goal_pos=(19, 19, 19)
        )
        
        obstacles = planner.generate_obstacles(
            num_obstacles=10,
            obstacle_types=['cube', 'sphere'],
            seed=123
        )
        
        self.assertEqual(len(obstacles), 10)
        self.assertGreater(len(planner.collision_detector.get_obstacle_hulls()), 0)
    
    def test_path_length_computation(self):
        """Test path length computation."""
        planner = PathPlanner(
            workspace_size=(10, 10, 10),
            start_pos=(0, 0, 0),
            goal_pos=(9, 9, 9)
        )
        
        path = planner.plan_path()
        
        if path:
            length = planner.get_path_length()
            self.assertGreater(length, 0)
            
            # Path length should be at least the straight-line distance
            straight_line_dist = np.sqrt(9**2 + 9**2 + 9**2)
            self.assertGreaterEqual(length, straight_line_dist * 0.9)  # Allow some tolerance
    
    def test_no_path_scenario(self):
        """Test scenario where no path exists."""
        planner = PathPlanner(
            workspace_size=(10, 10, 10),
            start_pos=(0, 0, 0),
            goal_pos=(9, 9, 9)
        )
        
        # Generate many large obstacles that might block the path
        obstacles = planner.generate_obstacles(
            num_obstacles=20,
            obstacle_types=['cube'],
            min_cube_size=3.0,
            max_cube_size=5.0,
            seed=999
        )
        
        path = planner.plan_path()
        
        # Path might be None or found depending on obstacle placement
        if path is None:
            # No path found - this is valid
            stats = planner.get_statistics()
            self.assertFalse(stats['path_found'])
            self.assertEqual(stats['path_length'], 0)
        else:
            # Path found - verify it's valid
            self.assertEqual(path[0], (0, 0, 0))
            self.assertEqual(path[-1], (9, 9, 9))


if __name__ == '__main__':
    unittest.main()

