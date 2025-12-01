"""
Data Export for Visualization

This module exports workspace, obstacles, hulls, and path data to JSON
for use in the Babylon.js visualization.
"""

import json
import os
from typing import Dict, Any
from path_planning.path_planner import PathPlanner


def export_path_data(path_planner: PathPlanner, output_file: str = "path_data.json"):
    """
    Export path planning data to JSON file for visualization.
    
    Args:
        path_planner: PathPlanner instance with computed path
        output_file: Output JSON file path
    """
    data = path_planner.export_for_visualization()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file


def export_path_data_to_string(path_planner: PathPlanner) -> str:
    """
    Export path planning data to JSON string.
    
    Args:
        path_planner: PathPlanner instance with computed path
    
    Returns:
        JSON string representation of the data
    """
    data = path_planner.export_for_visualization()
    return json.dumps(data, indent=2)

