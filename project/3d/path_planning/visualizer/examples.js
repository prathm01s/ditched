// Preloaded example scenarios for path planning

const examples = {
    "Simple Path": {
        workspace: { size: [20, 20, 20] },
        start: [0, 0, 0],
        goal: [19, 19, 19],
        num_obstacles: 3,
        obstacle_types: ["cube"],
        seed: 1
    },
    "Complex Maze": {
        workspace: { size: [20, 20, 20] },
        start: [0, 0, 0],
        goal: [19, 19, 19],
        num_obstacles: 15,
        obstacle_types: ["cube", "sphere"],
        seed: 42
    },
    "Narrow Passage": {
        workspace: { size: [20, 20, 20] },
        start: [0, 10, 0],
        goal: [19, 10, 19],
        num_obstacles: 12,
        obstacle_types: ["cube"],
        seed: 100
    },
    "High Obstacles": {
        workspace: { size: [20, 20, 20] },
        start: [0, 0, 0],
        goal: [19, 19, 19],
        num_obstacles: 8,
        obstacle_types: ["cube"],
        min_cube_size: 3.0,
        max_cube_size: 6.0,
        seed: 200
    },
    "Spherical Challenge": {
        workspace: { size: [20, 20, 20] },
        start: [0, 0, 0],
        goal: [19, 19, 19],
        num_obstacles: 10,
        obstacle_types: ["sphere"],
        min_sphere_radius: 1.5,
        max_sphere_radius: 3.5,
        seed: 300
    },
    "Mixed Environment": {
        workspace: { size: [20, 20, 20] },
        start: [0, 0, 0],
        goal: [19, 19, 19],
        num_obstacles: 12,
        obstacle_types: ["cube", "sphere"],
        min_cube_size: 2.0,
        max_cube_size: 5.0,
        min_sphere_radius: 1.0,
        max_sphere_radius: 3.0,
        seed: 500
    }
};

// Function to generate example data (would be called from Python backend)
// For now, we'll create a function that can be used to generate examples

