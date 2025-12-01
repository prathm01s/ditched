// 3D Robot Path Planning Visualization using Babylon.js

let scene, camera, engine;
let pathData = null;
let robotMesh = null;
let pathLine = null;
let obstacleMeshes = [];
let hullMeshes = [];
let gridMeshes = [];
let startMarker = null;
let goalMarker = null;
let animationRunning = false;
let animationSpeed = 1.0;
let currentPathIndex = 0;
let hullsGenerated = false;

// Initialize Babylon.js scene
function initScene() {
    const canvas = document.getElementById("renderCanvas");
    engine = new BABYLON.Engine(canvas, true);
    
    // Create scene with better background (matching main visualizer)
    scene = new BABYLON.Scene(engine);
    // Dark blue-purple background like main visualizer
    scene.clearColor = new BABYLON.Color3(0.1, 0.1, 0.15);
    scene.ambientColor = new BABYLON.Color3(0.3, 0.3, 0.35);
    
    // Define workspace center (default 20x20x20 workspace)
    const workspaceCenter = new BABYLON.Vector3(10, 10, 10);
    
    // Create camera
    camera = new BABYLON.ArcRotateCamera(
        "camera",
        -Math.PI / 2,
        Math.PI / 3,
        50,
        workspaceCenter.clone(),
        scene
    );
    // Attach camera controls - Babylon.js v8.x uses attachToCanvas
    try {
        // For Babylon.js v8.x - the method is attachToCanvas
        if (typeof camera.attachToCanvas === 'function') {
            camera.attachToCanvas(canvas, true);
            console.log("Camera controls attached using attachToCanvas");
        } 
        // For older versions
        else if (typeof camera.attachControls === 'function') {
            camera.attachControls(canvas, true);
            console.log("Camera controls attached using attachControls");
        }
        // Try inputs.attachElement for v8.x
        else if (camera.inputs && typeof camera.inputs.attachElement === 'function') {
            camera.inputs.attachElement(canvas);
            console.log("Camera controls attached using inputs.attachElement");
        }
        // Fallback - manual event handlers
        else {
            console.warn("Camera controls method not found, using manual controls");
            // Manual wheel zoom
            canvas.addEventListener("wheel", (e) => {
                e.preventDefault();
                camera.radius += e.deltaY * 0.01;
                camera.radius = Math.max(10, Math.min(200, camera.radius));
            });
        }
    } catch (error) {
        console.error("Error attaching camera controls:", error);
        // Continue without controls - scene will still render
    }
    camera.setTarget(workspaceCenter);
    camera.lowerRadiusLimit = 10;
    camera.upperRadiusLimit = 200;
    
    // Add better lighting
    const light1 = new BABYLON.HemisphericLight("light1", new BABYLON.Vector3(0, 1, 0), scene);
    light1.intensity = 0.8;
    light1.diffuse = new BABYLON.Color3(1, 1, 1);
    light1.specular = new BABYLON.Color3(0.5, 0.5, 0.5);
    
    const light2 = new BABYLON.DirectionalLight("light2", new BABYLON.Vector3(-1, -0.5, -1), scene);
    light2.intensity = 0.6;
    light2.diffuse = new BABYLON.Color3(1, 1, 0.95);
    light2.specular = new BABYLON.Color3(0.3, 0.3, 0.3);
    
    // Add point light for better visibility
    const light3 = new BABYLON.PointLight("light3", new BABYLON.Vector3(10, 15, 10), scene);
    light3.intensity = 0.3;
    light3.diffuse = new BABYLON.Color3(0.8, 0.8, 1);
    
    console.log("Lights added:", scene.lights.length);
    
    // Create grid helper
    createGrid();
    
    // Render loop
    engine.runRenderLoop(() => {
        scene.render();
    });
    
    // Handle window resize
    window.addEventListener("resize", () => {
        engine.resize();
    });
}

// Create grid visualization
function createGrid() {
    const gridSize = 20;
    gridMeshes = [];
    
    // Create grid lines on the ground plane (y=0)
    for (let i = 0; i <= gridSize; i += 2) { // Every 2 units to reduce clutter
        // X-axis lines (parallel to X, varying Z)
        const lineX = BABYLON.MeshBuilder.CreateLines("lineX" + i, {
            points: [
                new BABYLON.Vector3(0, 0, i),
                new BABYLON.Vector3(gridSize, 0, i)
            ],
            colors: [new BABYLON.Color4(0.4, 0.4, 0.4, 0.6), new BABYLON.Color4(0.4, 0.4, 0.4, 0.6)]
        }, scene);
        gridMeshes.push(lineX);
        
        // Z-axis lines (parallel to Z, varying X)
        const lineZ = BABYLON.MeshBuilder.CreateLines("lineZ" + i, {
            points: [
                new BABYLON.Vector3(i, 0, 0),
                new BABYLON.Vector3(i, 0, gridSize)
            ],
            colors: [new BABYLON.Color4(0.4, 0.4, 0.4, 0.6), new BABYLON.Color4(0.4, 0.4, 0.4, 0.6)]
        }, scene);
        gridMeshes.push(lineZ);
    }
    
    // Add axes for reference
    const axisLength = gridSize;
    // X axis (red)
    const axisX = BABYLON.MeshBuilder.CreateLines("axisX", {
        points: [
            new BABYLON.Vector3(0, 0, 0),
            new BABYLON.Vector3(axisLength, 0, 0)
        ],
        colors: [new BABYLON.Color4(1, 0, 0, 1), new BABYLON.Color4(1, 0, 0, 1)]
    }, scene);
    gridMeshes.push(axisX);
    
    // Y axis (green)
    const axisY = BABYLON.MeshBuilder.CreateLines("axisY", {
        points: [
            new BABYLON.Vector3(0, 0, 0),
            new BABYLON.Vector3(0, axisLength, 0)
        ],
        colors: [new BABYLON.Color4(0, 1, 0, 1), new BABYLON.Color4(0, 1, 0, 1)]
    }, scene);
    gridMeshes.push(axisY);
    
    // Z axis (blue)
    const axisZ = BABYLON.MeshBuilder.CreateLines("axisZ", {
        points: [
            new BABYLON.Vector3(0, 0, 0),
            new BABYLON.Vector3(0, 0, axisLength)
        ],
        colors: [new BABYLON.Color4(0, 0, 1, 1), new BABYLON.Color4(0, 0, 1, 1)]
    }, scene);
    gridMeshes.push(axisZ);
}

// Load path data from JSON
function loadPathData(data) {
    console.log("Loading path data:", data);
    pathData = data;
    clearScene();
    hullsGenerated = false;
    
    if (!pathData) {
        updateStatus("No data loaded");
        console.error("No path data provided");
        return;
    }
    
    try {
        // Visualize workspace
        visualizeWorkspace();
        
        // Visualize obstacles as POINTS only initially
        if (pathData.obstacles && pathData.obstacles.length > 0) {
            console.log("Visualizing obstacle points:", pathData.obstacles.length);
            visualizeObstaclePoints(pathData.obstacles);
        } else {
            console.warn("No obstacles in data");
        }
        
        // Don't visualize hulls initially - user must click Generate Hulls
        
        // Visualize start and goal
        if (pathData.start) {
            console.log("Visualizing start:", pathData.start);
            visualizeStart(pathData.start);
        }
        if (pathData.goal) {
            console.log("Visualizing goal:", pathData.goal);
            visualizeGoal(pathData.goal);
        }
        
        // Visualize path
        if (pathData.path && pathData.path.length > 0) {
            console.log("Visualizing path:", pathData.path.length, "nodes");
            visualizePath(pathData.path);
            updatePathInfo(pathData.path);
        } else {
            console.warn("No path in data");
        }
        
        // Create robot
        createRobot();
        
        updateStatus("Data loaded successfully");
        console.log("Visualization complete");
        
        // Update obstacle count
        updateObstacleCount(pathData.obstacles ? pathData.obstacles.length : 0);
        
        // Debug: Log scene info
        console.log("Scene meshes:", scene.meshes.length);
        console.log("Scene lights:", scene.lights.length);
        console.log("Camera position:", camera.position.toString());
        console.log("Camera target:", camera.getTarget().toString());
        console.log("Camera radius:", camera.radius);
        
        // Force camera to look at the scene
        if (pathData.path && pathData.path.length > 0) {
            const pathPoints = pathData.path.map(p => new BABYLON.Vector3(p[0], p[1], p[2]));
            const center = pathPoints.reduce((sum, p) => sum.add(p), new BABYLON.Vector3(0, 0, 0))
                .scale(1.0 / pathPoints.length);
            camera.setTarget(center);
            // Adjust camera radius to show entire path
            const maxDist = Math.max(...pathPoints.map(p => {
                const dist = BABYLON.Vector3.Distance(p, center);
                return dist;
            }));
            camera.radius = Math.max(40, maxDist * 2.5);
            console.log("Camera adjusted to focus on path center:", center.toString());
            console.log("Camera radius set to:", camera.radius);
        }
    } catch (error) {
        console.error("Error loading path data:", error);
        updateStatus("Error loading data: " + error.message);
    }
}

// Visualize workspace bounds
function visualizeWorkspace() {
    if (!pathData.workspace) return;
    
    const size = pathData.workspace.size;
    const workspaceCenter = new BABYLON.Vector3(size[0] / 2, size[1] / 2, size[2] / 2);
    
    // Create wireframe box for workspace
    const box = BABYLON.MeshBuilder.CreateBox("workspace", {
        width: size[0],
        height: size[1],
        depth: size[2]
    }, scene);
    box.position = workspaceCenter;
    const mat = new BABYLON.StandardMaterial("workspaceMat", scene);
    mat.wireframe = true;
    mat.emissiveColor = new BABYLON.Color3(0.3, 0.3, 0.3);
    mat.alpha = 0.3;
    box.material = mat;
    
    // Update camera to focus on workspace
    if (camera) {
        camera.setTarget(workspaceCenter);
        camera.radius = Math.max(size[0], size[1], size[2]) * 1.5;
    }
}

// Visualize obstacles as point clouds
function visualizeObstaclePoints(obstacles) {
    obstacleMeshes = [];
    
    obstacles.forEach((obstacle, index) => {
        if (!obstacle.points || obstacle.points.length === 0) {
            console.warn("Obstacle", index, "has no points");
            return;
        }
        
        // Create point cloud from obstacle points
        const positions = [];
        obstacle.points.forEach(point => {
            positions.push(point[0], point[1], point[2]);
        });
        
        // Create point cloud using sphere instances for better visibility
        const parentMesh = new BABYLON.Mesh("obstaclePoints" + index, scene);
        const pointTemplate = BABYLON.MeshBuilder.CreateSphere("pointTemplate", { diameter: 0.2, segments: 6 }, scene);
        pointTemplate.isVisible = false;
        
        const mat = new BABYLON.StandardMaterial("obstaclePointMat" + index, scene);
        mat.diffuseColor = new BABYLON.Color3(0.9, 0.3, 0.3);
        mat.emissiveColor = new BABYLON.Color3(0.3, 0.1, 0.1);
        
        for (let i = 0; i < obstacle.points.length; i++) {
            const point = obstacle.points[i];
            const instance = pointTemplate.createInstance("point" + index + "_" + i);
            instance.position = new BABYLON.Vector3(point[0], point[1], point[2]);
            instance.parent = parentMesh;
            instance.material = mat;
        }
        
        obstacleMeshes.push(parentMesh);
        console.log(`Obstacle ${index}: ${obstacle.points.length} points`);
    });
}

// Visualize obstacles (old function - kept for compatibility)
function visualizeObstacles(obstacles) {
    obstacleMeshes = [];
    
    obstacles.forEach((obstacle, index) => {
        if (obstacle.type === 'cube') {
            const bounds = obstacle.bounds;
            const width = bounds.x[1] - bounds.x[0];
            const height = bounds.y[1] - bounds.y[0];
            const depth = bounds.z[1] - bounds.z[0];
            const center = obstacle.center || [
                (bounds.x[0] + bounds.x[1]) / 2,
                (bounds.y[0] + bounds.y[1]) / 2,
                (bounds.z[0] + bounds.z[1]) / 2
            ];
            
            const box = BABYLON.MeshBuilder.CreateBox("obstacle" + index, {
                width: width,
                height: height,
                depth: depth
            }, scene);
            box.position = new BABYLON.Vector3(center[0], center[1], center[2]);
            
            const mat = new BABYLON.StandardMaterial("obstacleMat" + index, scene);
            mat.diffuseColor = new BABYLON.Color3(0.9, 0.3, 0.3);
            mat.emissiveColor = new BABYLON.Color3(0.2, 0.05, 0.05);
            mat.specularColor = new BABYLON.Color3(0.5, 0.2, 0.2);
            mat.alpha = 0.7;
            box.material = mat;
            
            obstacleMeshes.push(box);
        } else if (obstacle.type === 'sphere') {
            const center = obstacle.center || [0, 0, 0];
            const radius = obstacle.bounds ? 
                Math.max(
                    (obstacle.bounds.x[1] - obstacle.bounds.x[0]) / 2,
                    (obstacle.bounds.y[1] - obstacle.bounds.y[0]) / 2,
                    (obstacle.bounds.z[1] - obstacle.bounds.z[0]) / 2
                ) : 1.0;
            
            const sphere = BABYLON.MeshBuilder.CreateSphere("obstacle" + index, {
                diameter: radius * 2
            }, scene);
            sphere.position = new BABYLON.Vector3(center[0], center[1], center[2]);
            
            const mat = new BABYLON.StandardMaterial("obstacleMat" + index, scene);
            mat.diffuseColor = new BABYLON.Color3(0.9, 0.3, 0.3);
            mat.emissiveColor = new BABYLON.Color3(0.2, 0.05, 0.05);
            mat.specularColor = new BABYLON.Color3(0.5, 0.2, 0.2);
            mat.alpha = 0.7;
            sphere.material = mat;
            
            obstacleMeshes.push(sphere);
        }
    });
}

// Visualize convex hulls using SubMesh/MultiMaterial approach
function visualizeHulls(hulls) {
    hullMeshes = [];
    
    console.log("Visualizing", hulls.length, "hulls");
    
    hulls.forEach((hull, hullIndex) => {
        if (!hull.vertices || !hull.faces) {
            console.warn("Hull", hullIndex, "missing vertices or faces");
            return;
        }
        
        console.log(`Hull ${hullIndex}: ${hull.vertices.length} vertices, ${hull.faces.length} faces`);
        
        // Build vertices array - each face gets its own vertices for proper normals
        const vertices = [];
        const faces = [];
        const normals = [];
        
        hull.faces.forEach((face, faceIdx) => {
            if (face.length < 3) {
                console.warn(`Hull ${hullIndex} face ${faceIdx} has less than 3 vertices`);
                return;
            }
            
            // Get the actual vertex positions for this face
            const faceVerts = face.map(idx => hull.vertices[idx]);
            
            // Compute face normal
            const v0 = new BABYLON.Vector3(faceVerts[0][0], faceVerts[0][1], faceVerts[0][2]);
            const v1 = new BABYLON.Vector3(faceVerts[1][0], faceVerts[1][1], faceVerts[1][2]);
            const v2 = new BABYLON.Vector3(faceVerts[2][0], faceVerts[2][1], faceVerts[2][2]);
            
            const edge1 = v1.subtract(v0);
            const edge2 = v2.subtract(v0);
            const normal = BABYLON.Vector3.Cross(edge1, edge2).normalize();
            
            // Add vertices for this face (triangulated)
            for (let i = 0; i < 3; i++) {
                vertices.push(faceVerts[i][0], faceVerts[i][1], faceVerts[i][2]);
                normals.push(normal.x, normal.y, normal.z);
            }
            const baseIdx = faces.length;
            faces.push(baseIdx, baseIdx + 1, baseIdx + 2);
            
            // If face has more than 3 vertices, triangulate (fan triangulation)
            if (faceVerts.length > 3) {
                for (let i = 3; i < faceVerts.length; i++) {
                    // Add triangle: v0, v[i-1], v[i]
                    vertices.push(faceVerts[0][0], faceVerts[0][1], faceVerts[0][2]);
                    vertices.push(faceVerts[i-1][0], faceVerts[i-1][1], faceVerts[i-1][2]);
                    vertices.push(faceVerts[i][0], faceVerts[i][1], faceVerts[i][2]);
                    
                    normals.push(normal.x, normal.y, normal.z);
                    normals.push(normal.x, normal.y, normal.z);
                    normals.push(normal.x, normal.y, normal.z);
                    
                    const idx = faces.length;
                    faces.push(idx, idx + 1, idx + 2);
                }
            }
        });
        
        if (faces.length === 0) {
            console.warn("Hull", hullIndex, "has no valid triangulated faces");
            return;
        }
        
        // Create mesh
        const hullMesh = new BABYLON.Mesh("hull" + hullIndex, scene);
        const vertexData = new BABYLON.VertexData();
        
        vertexData.positions = vertices;
        vertexData.indices = faces;
        vertexData.normals = normals;
        
        vertexData.applyToMesh(hullMesh);
        
        // Create MultiMaterial with one material per face
        const multiMaterial = new BABYLON.MultiMaterial("hullMultiMat" + hullIndex, scene);
        hullMesh.subMeshes = [];
        
        let triangleOffset = 0;
        hull.faces.forEach((face, faceIdx) => {
            // Calculate how many triangles this face generated
            const numTriangles = face.length - 2;
            
            // Create material for this face with random color
            const mat = new BABYLON.StandardMaterial(`hullMat${hullIndex}_face${faceIdx}`, scene);
            mat.backFaceCulling = false;
            
            // Random color per face
            const color = new BABYLON.Color3(Math.random(), Math.random(), Math.random());
            mat.diffuseColor = color;
            mat.alpha = 0.5;
            mat.specularPower = 64;
            
            multiMaterial.subMaterials.push(mat);
            
            // Create submesh for this face's triangles
            new BABYLON.SubMesh(
                faceIdx,
                0,
                vertices.length / 3,
                triangleOffset * 3,
                numTriangles * 3,
                hullMesh
            );
            
            triangleOffset += numTriangles;
        });
        
        hullMesh.material = multiMaterial;
        hullMeshes.push(hullMesh);
        
        console.log(`Hull ${hullIndex} created with ${vertices.length / 3} vertices, ${faces.length / 3} triangles, ${hull.faces.length} faces`);
    });
}

// Helper function to convert HSL to RGB
function HSLToRGB(h, s, l) {
    s /= 100;
    l /= 100;
    const k = n => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return {
        r: f(0),
        g: f(8),
        b: f(4)
    };
}

// Visualize start position
function visualizeStart(position) {
    if (startMarker) startMarker.dispose();
    
    // Make start marker larger and more visible
    startMarker = BABYLON.MeshBuilder.CreateSphere("start", { diameter: 1.5, segments: 16 }, scene);
    startMarker.position = new BABYLON.Vector3(position[0], position[1], position[2]);
    
    const mat = new BABYLON.StandardMaterial("startMat", scene);
    mat.diffuseColor = new BABYLON.Color3(0, 1, 0);
    mat.emissiveColor = new BABYLON.Color3(0, 0.8, 0);
    mat.specularColor = new BABYLON.Color3(0, 1, 0);
    startMarker.material = mat;
    
    console.log("Start marker at:", position);
}

// Visualize goal position
function visualizeGoal(position) {
    if (goalMarker) goalMarker.dispose();
    
    // Make goal marker larger and more visible
    goalMarker = BABYLON.MeshBuilder.CreateSphere("goal", { diameter: 1.5, segments: 16 }, scene);
    goalMarker.position = new BABYLON.Vector3(position[0], position[1], position[2]);
    
    const mat = new BABYLON.StandardMaterial("goalMat", scene);
    mat.diffuseColor = new BABYLON.Color3(1, 0, 0);
    mat.emissiveColor = new BABYLON.Color3(0.8, 0, 0);
    mat.specularColor = new BABYLON.Color3(1, 0, 0);
    goalMarker.material = mat;
    
    console.log("Goal marker at:", position);
}

// Visualize path
function visualizePath(path) {
    if (pathLine) {
        pathLine.dispose();
        pathLine = null;
    }
    
    if (!path || path.length < 2) {
        console.warn("Path is empty or has less than 2 points");
        return;
    }
    
    try {
        const points = path.map(p => {
            if (!Array.isArray(p) || p.length < 3) {
                console.warn("Invalid path point:", p);
                return new BABYLON.Vector3(0, 0, 0);
            }
            return new BABYLON.Vector3(p[0], p[1], p[2]);
        });
        
        // Create thicker, more visible path line with gradient colors
        const pathColors = points.map((p, i) => {
            // Gradient from green (start) to yellow (middle) to red (end)
            const t = i / (points.length - 1);
            if (t < 0.5) {
                // Green to yellow
                const localT = t * 2;
                return new BABYLON.Color4(0, 1 - localT * 0.5, localT, 1);
            } else {
                // Yellow to red
                const localT = (t - 0.5) * 2;
                return new BABYLON.Color4(localT, 1 - localT * 0.5, 0, 1);
            }
        });
        
        pathLine = BABYLON.MeshBuilder.CreateLines("path", {
            points: points,
            colors: pathColors,
            updatable: false
        }, scene);
        
        // Make path more visible
        if (pathLine.color) {
            pathLine.color = new BABYLON.Color3(1, 1, 0);
        }
        
        console.log("Path visualized with", points.length, "points");
        console.log("Path bounds:", {
            min: points.reduce((min, p) => new BABYLON.Vector3(
                Math.min(min.x, p.x), Math.min(min.y, p.y), Math.min(min.z, p.z)
            ), points[0]),
            max: points.reduce((max, p) => new BABYLON.Vector3(
                Math.max(max.x, p.x), Math.max(max.y, p.y), Math.max(max.z, p.z)
            ), points[0])
        });
    } catch (error) {
        console.error("Error visualizing path:", error);
    }
}

// Create robot mesh
function createRobot() {
    if (robotMesh) robotMesh.dispose();
    
    // Make robot more visible and appealing
    robotMesh = BABYLON.MeshBuilder.CreateSphere("robot", { diameter: 0.8, segments: 16 }, scene);
    
    const mat = new BABYLON.StandardMaterial("robotMat", scene);
    mat.diffuseColor = new BABYLON.Color3(0, 0.9, 1);
    mat.emissiveColor = new BABYLON.Color3(0, 0.5, 0.6);
    mat.specularColor = new BABYLON.Color3(0.5, 0.9, 1);
    robotMesh.material = mat;
    
    // Position robot at start
    if (pathData && pathData.path && pathData.path.length > 0) {
        const start = pathData.path[0];
        robotMesh.position = new BABYLON.Vector3(start[0], start[1], start[2]);
    }
}

// Start animation
function startAnimation() {
    if (!pathData || !pathData.path || pathData.path.length < 2) {
        updateStatus("No path to animate");
        return;
    }
    
    animationRunning = true;
    currentPathIndex = 0;
    
    document.getElementById("playBtn").disabled = true;
    document.getElementById("stopBtn").disabled = false;
    
    updateStatus("Animating...");
    
    animateRobot();
}

// Stop animation
function stopAnimation() {
    animationRunning = false;
    document.getElementById("playBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
    updateStatus("Animation stopped");
}

// Reset animation
function resetAnimation() {
    stopAnimation();
    currentPathIndex = 0;
    if (robotMesh && pathData && pathData.path && pathData.path.length > 0) {
        const start = pathData.path[0];
        robotMesh.position = new BABYLON.Vector3(start[0], start[1], start[2]);
    }
    updateStatus("Animation reset");
}

// Animate robot along path
function animateRobot() {
    if (!animationRunning || !pathData || !pathData.path) return;
    
    if (currentPathIndex >= pathData.path.length - 1) {
        stopAnimation();
        updateStatus("Animation complete");
        return;
    }
    
    const current = pathData.path[currentPathIndex];
    const next = pathData.path[currentPathIndex + 1];
    
    const startPos = new BABYLON.Vector3(current[0], current[1], current[2]);
    const endPos = new BABYLON.Vector3(next[0], next[1], next[2]);
    
    const distance = BABYLON.Vector3.Distance(startPos, endPos);
    const duration = (distance / 2.0) / animationSpeed; // Base speed: 2 units per second
    
    let elapsed = 0;
    const startTime = Date.now();
    
    const animate = () => {
        if (!animationRunning) return;
        
        elapsed = (Date.now() - startTime) / 1000;
        const t = Math.min(elapsed / duration, 1.0);
        
        robotMesh.position = BABYLON.Vector3.Lerp(startPos, endPos, t);
        
        if (t >= 1.0) {
            currentPathIndex++;
            if (currentPathIndex < pathData.path.length - 1) {
                setTimeout(animateRobot, 0);
            } else {
                stopAnimation();
                updateStatus("Animation complete");
            }
        } else {
            requestAnimationFrame(animate);
        }
    };
    
    animate();
}

// Update animation speed
function updateSpeed(value) {
    animationSpeed = parseFloat(value);
    document.getElementById("speedValue").textContent = value + "x";
}

// Toggle path visibility
function togglePath(show) {
    if (pathLine) pathLine.setEnabled(show);
}

// Toggle hulls visibility
function toggleHulls(show) {
    hullMeshes.forEach(mesh => mesh.setEnabled(show));
}

// Toggle obstacles visibility
function toggleObstacles(show) {
    obstacleMeshes.forEach(mesh => mesh.setEnabled(show));
}

// Toggle grid visibility
function toggleGrid(show) {
    gridMeshes.forEach(mesh => mesh.setEnabled(show));
}

// Clear scene
function clearScene() {
    obstacleMeshes.forEach(mesh => mesh.dispose());
    obstacleMeshes = [];
    
    hullMeshes.forEach(mesh => mesh.dispose());
    hullMeshes = [];
    
    if (pathLine) pathLine.dispose();
    pathLine = null;
    
    if (startMarker) startMarker.dispose();
    startMarker = null;
    
    if (goalMarker) goalMarker.dispose();
    goalMarker = null;
    
    if (robotMesh) robotMesh.dispose();
    robotMesh = null;
}

// Update status
function updateStatus(message) {
    document.getElementById("status").textContent = message;
}

// Update path info
function updatePathInfo(path) {
    if (!path || path.length < 2) {
        document.getElementById("pathLength").textContent = "-";
        document.getElementById("pathNodes").textContent = "-";
        return;
    }
    
    // Calculate path length
    let length = 0;
    for (let i = 0; i < path.length - 1; i++) {
        const p1 = path[i];
        const p2 = path[i + 1];
        const dx = p2[0] - p1[0];
        const dy = p2[1] - p1[1];
        const dz = p2[2] - p1[2];
        length += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    
    document.getElementById("pathLength").textContent = length.toFixed(2);
    document.getElementById("pathNodes").textContent = path.length;
}

// Load data from file
function loadDataFile() {
    const fileInput = document.getElementById("dataFile");
    const file = fileInput.files[0];
    
    if (!file) {
        alert("Please select a JSON file");
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            console.log("Loaded data:", data);
            loadPathData(data);
        } catch (error) {
            console.error("Error parsing JSON:", error);
            alert("Error parsing JSON file: " + error.message);
        }
    };
    reader.onerror = (error) => {
        console.error("Error reading file:", error);
        alert("Error reading file");
    };
    reader.readAsText(file);
}

// Load example scenario from pre-generated files
async function loadExample(exampleName) {
    updateStatus("Loading example...");
    
    try {
        const response = await fetch(`examples/${exampleName}.json`);
        if (response.ok) {
            const data = await response.json();
            loadPathData(data);
            updateStatus(`Loaded: ${exampleName}`);
        } else {
            // Fallback: try to load from current directory
            const response2 = await fetch(`${exampleName}.json`);
            if (response2.ok) {
                const data = await response2.json();
                loadPathData(data);
                updateStatus(`Loaded: ${exampleName}`);
            } else {
                updateStatus("Example file not found. Generate examples first.");
                alert(`Example file not found: ${exampleName}.json\n\n` +
                      `Please run: python path_planning/generate_examples.py\n\n` +
                      `Or generate manually using demo.py`);
            }
        }
    } catch (error) {
        console.error("Error loading example:", error);
        updateStatus("Error loading example");
        alert(`Error loading example: ${error.message}\n\n` +
              `Please run: python path_planning/generate_examples.py to generate example files.`);
    }
}

// Update obstacle count in info panel
function updateObstacleCount(count) {
    const elem = document.getElementById("obstacleCount");
    if (elem) elem.textContent = count || "-";
}

// Generate hulls from obstacle points
function generateHulls() {
    if (!pathData || !pathData.hulls || pathData.hulls.length === 0) {
        updateStatus("No hull data available");
        alert("No hull data in the current scene. Please load a scenario first.");
        return;
    }
    
    if (hullsGenerated) {
        updateStatus("Hulls already generated");
        return;
    }
    
    updateStatus("Generating hulls...");
    visualizeHulls(pathData.hulls);
    hullsGenerated = true;
    updateStatus("Hulls generated");
}

// Clear all hulls from the scene
function clearHulls() {
    hullMeshes.forEach(mesh => {
        if (mesh) mesh.dispose();
    });
    hullMeshes = [];
    hullsGenerated = false;
    updateStatus("Hulls cleared");
}

// Initialize scene when page loads
window.addEventListener("DOMContentLoaded", () => {
    initScene();
});

