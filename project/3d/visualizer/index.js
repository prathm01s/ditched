import { CONSTS } from './consts.js';
import {Quickhull3D} from './Quickhull3D.js'
import {JarvisMarch3D} from './JarvisMarch3D.js'
import {IncrementalHull3D} from './IncrementalHull3D.js'
import {Face, FaceTypes} from './Face.js';

let ID_COUNTER = 0;

function showWorldAxis(size, scene) {
    var makeTextPlane = function(text, color, size) {
        var dynamicTexture = new BABYLON.DynamicTexture("DynamicTexture", 50, scene, true);
        dynamicTexture.hasAlpha = true;
        dynamicTexture.drawText(text, 5, 40, "bold 36px Arial", color , "transparent", true);
        var plane = BABYLON.Mesh.CreatePlane("TextPlane", size, scene, true);
        plane.material = new BABYLON.StandardMaterial("TextPlaneMaterial", scene);
        plane.material.backFaceCulling = false;
        plane.material.specularColor = new BABYLON.Color3(0, 0, 0);
        plane.material.diffuseTexture = dynamicTexture;
    return plane;
     };
    var axisX = BABYLON.Mesh.CreateLines("axisX", [ 
      BABYLON.Vector3.Zero(), new BABYLON.Vector3(size, 0, 0), new BABYLON.Vector3(size * 0.95, 0.05 * size, 0), 
      new BABYLON.Vector3(size, 0, 0), new BABYLON.Vector3(size * 0.95, -0.05 * size, 0)
      ], scene);
    axisX.color = new BABYLON.Color3(1, 0, 0);
    var xChar = makeTextPlane("X", "red", size / 10);
    xChar.position = new BABYLON.Vector3(0.9 * size, -0.05 * size, 0);
    var axisY = BABYLON.Mesh.CreateLines("axisY", [
        BABYLON.Vector3.Zero(), new BABYLON.Vector3(0, size, 0), new BABYLON.Vector3( -0.05 * size, size * 0.95, 0), 
        new BABYLON.Vector3(0, size, 0), new BABYLON.Vector3( 0.05 * size, size * 0.95, 0)
        ], scene);
    axisY.color = new BABYLON.Color3(0, 1, 0);
    var yChar = makeTextPlane("Y", "green", size / 10);
    yChar.position = new BABYLON.Vector3(0, 0.9 * size, -0.05 * size);
    var axisZ = BABYLON.Mesh.CreateLines("axisZ", [
        BABYLON.Vector3.Zero(), new BABYLON.Vector3(0, 0, size), new BABYLON.Vector3( 0 , -0.05 * size, size * 0.95),
        new BABYLON.Vector3(0, 0, size), new BABYLON.Vector3( 0, 0.05 * size, size * 0.95)
        ], scene);
    axisZ.color = new BABYLON.Color3(0, 0, 1);
    var zChar = makeTextPlane("Z", "blue", size / 10);
    zChar.position = new BABYLON.Vector3(0, 0.05 * size, 0.9 * size);
};

function createScene(engine, canvas) {
    // This creates a basic Babylon Scene object (non-mesh)
    var scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color3(0.1, 0.1, 0.15); // Dark background

    // This creates and positions a free camera (non-mesh)
    var camera = new BABYLON.ArcRotateCamera(
        "camera1", 
        1.3*Math.PI/2, 
        1.2*Math.PI/4, 
        100,  // Increased distance to see more
        new BABYLON.Vector3(0,0,0), 
        scene
    );

    // This targets the camera to scene origin
    camera.setTarget(BABYLON.Vector3.Zero());
    
    // Set camera limits
    camera.lowerRadiusLimit = 10;
    camera.upperRadiusLimit = 200;

    // This attaches the camera to the canvas
    camera.attachControl(canvas, true);

    // This creates a light, aiming 0,1,0 - to the sky (non-mesh)
    var light = new BABYLON.HemisphericLight("light1", new BABYLON.Vector3(0, -1, 0), scene);
    var light2 = new BABYLON.HemisphericLight("light2", new BABYLON.Vector3(0, 1, 0), scene);
    var light3 = new BABYLON.DirectionalLight("light3", new BABYLON.Vector3(-1, -1, -1), scene);

    // Default intensity is 1. Let's dim the light a small amount
    light.intensity = 0.7;
    light2.intensity = 0.4;
    light3.intensity = 0.6;

    return scene;
}

// https://playground.babylonjs.com/#X7U2LD#1 reference of how to update the faces data in real time
function constructHull(inputPoints, algorithm = 'quickhull') {
    if (algorithm === 'jarvis') {
        const jm = new JarvisMarch3D(inputPoints, 0);
        jm.compute();
        return jm;
    } else if (algorithm === 'incremental') {
        const hull = new IncrementalHull3D();
        hull.build(inputPoints);
        return hull;
    } else {
        // Default to QuickHull
        const hull = new Quickhull3D();
        hull.build(inputPoints);
        return hull;
    }
}

function rndOneMinusOne() {
    return Math.random() * 2 - 1;
}

function parsePointsFromText(text) {
    const points = [];
    const lines = text.split('\n');
    
    for (let line of lines) {
        line = line.trim();
        // Skip empty lines and comments
        if (!line || line.startsWith('#')) {
            continue;
        }
        
        // Handle lines with = (assignment) - skip them
        if (line.includes('=')) {
            continue;
        }
        
        const parts = line.split(/\s+/);
        if (parts.length >= 3) {
            try {
                const x = parseFloat(parts[0]);
                const y = parseFloat(parts[1]);
                const z = parseFloat(parts[2]);
                
                if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
                    const v = new BABYLON.Vector3(x, y, z);
                    v.id = ID_COUNTER++;
                    points.push(v);
                }
            } catch (e) {
                console.warn('Failed to parse line:', line, e);
            }
        }
    }
    
    return points;
}

function createGroupFromPoints(points, color, scene) {
    if (!points || points.length === 0) {
        return null;
    }
    
    // Assign IDs to points if they don't have them
    points.forEach((p, i) => {
        if (!p.id) {
            p.id = ID_COUNTER++;
        }
    });
    
    const struct = {
        points: points,
        renderPos: new BABYLON.Vector3(0, 0, 0),
        renderRot: new BABYLON.Vector3(0, 0, 0),
        singleCol: color || new BABYLON.Color3(0.8, 0.8, 1.0)
    };
    
    pointsToPcs(points, struct.renderPos, struct.renderRot, struct.singleCol, struct);
    
    console.log(`Created group with ${points.length} points`);
    
    return struct;
}

function clearScene(scene) {
    // Collect all meshes to remove (except axes and UI elements)
    const meshesToRemove = [];
    const meshNamesToKeep = ["axisX", "axisY", "axisZ", "renderCanvas"];
    
    // Get all meshes
    const allMeshes = scene.meshes.slice(); // Copy array to avoid modification during iteration
    
    allMeshes.forEach(m => {
        const name = m.name || "";
        const shouldKeep = meshNamesToKeep.some(keepName => name === keepName) ||
                          name.startsWith("TextPlane");
        
        if (!shouldKeep) {
            meshesToRemove.push(m);
        }
    });
    
    // Dispose all meshes
    meshesToRemove.forEach(m => {
        try {
            if (m.dispose) {
                m.dispose();
            }
        } catch (e) {
            console.warn("Error disposing mesh:", m.name, e);
        }
    });
    
    // Clear particle systems
    scene.particleSystems.forEach(ps => {
        try {
            if (ps.dispose) {
                ps.dispose();
            }
        } catch (e) {
            console.warn("Error disposing particle system:", e);
        }
    });
    
    // Clear any remaining geometry
    scene.geometries.forEach(geo => {
        try {
            if (geo.dispose) {
                geo.dispose();
            }
        } catch (e) {
            // Ignore disposal errors
        }
    });
    
    console.log(`Cleared ${meshesToRemove.length} meshes from scene`);
    return [];
}

function updateStatus(message, type = 'info') {
    const statusBar = document.getElementById("statusBar");
    const statusText = document.getElementById("statusText");
    
    if (statusBar && statusText) {
        // Remove previous status classes
        statusBar.classList.remove('success', 'error', 'warning');
        
        // Add appropriate class
        if (type === 'success' || message.startsWith('✓')) {
            statusBar.classList.add('success');
        } else if (type === 'error' || message.startsWith('✗')) {
            statusBar.classList.add('error');
        } else if (type === 'warning' || message.startsWith('⚠')) {
            statusBar.classList.add('warning');
        }
        
        statusText.textContent = message;
    }
    console.log(message);
}

async function loadSampleFile(filename) {
    try {
        // Try multiple possible paths
        const paths = [
            `../${filename}`,
            `../../${filename}`,
            `./${filename}`,
            filename
        ];
        
        let response = null;
        let lastError = null;
        
        for (const path of paths) {
            try {
                response = await fetch(path);
                if (response.ok) {
                    break;
                }
            } catch (e) {
                lastError = e;
                continue;
            }
        }
        
        if (!response || !response.ok) {
            throw new Error(`Failed to load ${filename} from any path. Last error: ${lastError?.message || response?.statusText}`);
        }
        
        const text = await response.text();
        const points = parsePointsFromText(text);
        if (points.length < 4) {
            updateStatus(`✗ Error: ${filename} has less than 4 points (got ${points.length})`, 'error');
            return null;
        }
        return points;
    } catch (error) {
        console.error(`Error loading ${filename}:`, error);
        updateStatus(`✗ Error loading ${filename}: ${error.message}`, 'error');
        return null;
    }
}

function pointsToPcs(testGroup, renderPos, renderRot, singleCol, struct) {
    if (!testGroup || testGroup.length === 0) {
        console.warn("pointsToPcs: Empty test group");
        return;
    }
    
    let visPcs = new BABYLON.PointsCloudSystem("pcs-" + Math.random().toString(36).substr(2, 9), 2);
    visPcs.addPoints(testGroup.length, (p, i, s) => {
        if (testGroup[i]) {
        p.position = testGroup[i];
            p.color = singleCol || new BABYLON.Color3(1, 1, 1);
        }
    });
    visPcs.buildMeshAsync().then((mesh) => {
        if (mesh) {
        mesh.position = renderPos;
        mesh.rotation = renderRot;
        struct.pointsMesh = mesh;
            console.log(`Created point cloud with ${testGroup.length} points`);
        }
    }).catch((error) => {
        console.error("Error building point cloud:", error);
    });
}

function makeCylinderGroup(nSubdiv, height, radius, renderPos, renderRot, singleCol, extraVertices, topGroup, bottomGroup) {
    let testGroup = [];
    // console.log('cylinder group renderpos', renderPos);
    let tot = nSubdiv;
    let quat = BABYLON.Quaternion.FromEulerAngles(renderRot.x, renderRot.y, renderRot.z);
    // console.log('quat', quat);
    // console.log('top group', topGroup);
    // console.log('bot group', bottomGroup);
    for (let i = 1; i < tot+1; i++) {
        let vtop = new BABYLON.Vector3(Math.sin(2*Math.PI*(i/tot))*radius, height/2, Math.cos(2*Math.PI*(i/tot))*radius);
        vtop.id = ID_COUNTER++;
        // console.log('vtop before rotate', vtop);
        vtop.rotateByQuaternionToRef(quat, vtop);
        // console.log('vtop after rotate', vtop);
        vtop = vtop.add(renderPos);
        testGroup.push( vtop );
        topGroup.push(vtop);
        let vbot = new BABYLON.Vector3(Math.sin(2*Math.PI*(i/tot))*radius, -height/2, Math.cos(2*Math.PI*(i/tot))*radius);
        vbot.id = ID_COUNTER++;
        vbot.rotateByQuaternionToRef(quat, vbot);
        vbot = vbot.add(renderPos);
        testGroup.push( vbot );
        bottomGroup.push(vbot);
    }

    // Add some percentage of random points inside
    let rndInside = testGroup.length * Math.random();
    for (let i = 0; i < rndInside; i++) {
        let a = 2*Math.PI * Math.random();
        let r = radius * Math.random();

        let x = Math.sin(a)*r;
        let y = rndOneMinusOne() * height/2; 
        let z = Math.cos(a)*r;

        let vi = new BABYLON.Vector3(x,y,z);
        vi.id = ID_COUNTER++;
        vi.rotateByQuaternionToRef(quat, vi);
        vi = vi.add(renderPos);

        if (vi.equalsWithEpsilon(new BABYLON.Vector3(0,0,0))) {
            console.log('got rnd close to 0');
        }

        testGroup.push(vi);
    }

    testGroup.push(...extraVertices);

    // console.log('cylinder test group', testGroup);

    let struct = {
        points: testGroup,
        renderPos: new BABYLON.Vector3(0,0,0),
        renderRot: new BABYLON.Vector3(0,0,0),
        singleCol
    }

    pointsToPcs(testGroup, struct.renderPos, struct.renderRot, singleCol, struct);

    return struct;
}

function addAndRotRef(v, add, rot) {
    v.rotateByQuaternionToRef(rot, v).addToRef(add, v);
}

function makeBoxGroup(npts, width, height, depth, renderPos, renderRot, singleCol, extraVertices) {
    let testGroup = [];

    let quat = BABYLON.Quaternion.FromEulerAngles(renderRot.x, renderRot.y, renderRot.z);
    for (let i = 0; i < npts; i++) {
        let a = new BABYLON.Vector3(width*i/npts-width*0.5, -height*0.5, -depth*0.5);
        addAndRotRef(a, renderPos, quat);
        testGroup.push(a);
        let b = new BABYLON.Vector3(width*i/npts-width*0.5, height*0.5, -depth*0.5);
        addAndRotRef(b, renderPos, quat);
        testGroup.push(b);
        let c = new BABYLON.Vector3(width*i/npts-width*0.5, height*0.5, depth*0.5)
        addAndRotRef(c, renderPos, quat);
        testGroup.push(c);
        let d = new BABYLON.Vector3(width*i/npts-width*0.5, -height*0.5, depth*0.5);
        addAndRotRef(d, renderPos, quat);
        testGroup.push(d);
    }

    for (let i = 0; i < npts; i++) {
        let a = new BABYLON.Vector3(rndOneMinusOne()*width/4, rndOneMinusOne()*height/4, rndOneMinusOne()*depth/4);
        addAndRotRef(a, renderPos, quat);    
        testGroup.push(a);
    }

    testGroup.push(...extraVertices);

    let struct = {
        points: testGroup,
        renderPos: new BABYLON.Vector3(0,0,0),
        renderRot: new BABYLON.Vector3(0,0,0),
        singleCol
    }

    pointsToPcs(testGroup, struct.renderPos, struct.renderRot, singleCol, struct);
    
    return struct;
}

function makeSphereGroup(nPts, radius, renderPos, singleCol, verticesUnderHeight, verticesUnderHeightGroup) {
    let testGroup = [];
    // console.log('vertices under height', verticesUnderHeight);
    const phi = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < nPts; i++) {
        const y = 1 - (i / (nPts-1)) * 2;
        const inRadius = Math.sqrt(1 - y*y);
        
        const theta = phi * i;
        const x = Math.cos(theta)*inRadius;
        const z = Math.sin(theta)*inRadius;
        
        const v = new BABYLON.Vector3(x,y,z).scale(radius).add(renderPos);
        v.id = ID_COUNTER++;

        testGroup.push(v);
        // console.log('y radius of', y*radius);
        if (y*radius < verticesUnderHeight) {
            verticesUnderHeightGroup.push(v);
        }
    }

    let struct = {
        points: testGroup,
        renderPos: new BABYLON.Vector3(0,0,0),
        renderRot: new BABYLON.Vector3(0,0,0),
        singleCol
    }

    pointsToPcs(testGroup, struct.renderPos, struct.renderRot, singleCol, struct);

    return struct;
}

function buildThemeGroups() {
    const groups = [];
    
    // const sph_sub = 80;
    const sph_sub = 70;
    // const cyl_sub = 20;
    const cyl_sub = 19;
    const mid_t_height = 7;
    const mid_t_radius = 6;
    const ul_t_radius = 5;
    const ul_t_height = 2;
    const h_rad = 6;
    const leg_len = 5;
    const leg_rad = 1.5;
    const leg_x_pos = 6;
    const arm_len = 11;
    const arm_rad = 1;
    const foot_w = 6;
    const foot_h = 1;
    const foot_d = 4;
    const foot_pts = 3;
    const foot_x_pos = 1;
    const hand_w = 3;
    const hand_h = 4;
    const hand_d = 2;
    const hand_pts = 2;
    
    const head_connection = [];
    // head
    groups.push(
        makeSphereGroup(
            sph_sub, 
            h_rad, 
            new BABYLON.Vector3(0, ul_t_height + mid_t_height/2 + h_rad, 0), 
            new BABYLON.Color3(1,1,0), 
            -h_rad*0.8, 
            head_connection)); 
    
    const left_arm_torso_connection = [];
    const left_arm_hand_connection = [];
    // left arm
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            arm_len, 
            arm_rad, 
            new BABYLON.Vector3(mid_t_radius*1.8, ul_t_height + mid_t_height/2, 0), 
            new BABYLON.Vector3(0,0,-Math.PI/2), 
            new BABYLON.Color3(1,0,1),
            [],
            left_arm_hand_connection,
            left_arm_torso_connection));
    // right arm
    const right_arm_torso_connection = [];
    const right_arm_hand_connection = [];
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            arm_len, 
            arm_rad, 
            new BABYLON.Vector3(-mid_t_radius*1.8, ul_t_height + mid_t_height/2, 0), 
            new BABYLON.Vector3(0,0,Math.PI/2), 
            new BABYLON.Color3(1,0,1),
            [],
            right_arm_hand_connection,
            right_arm_torso_connection));
    
    const upper_torso_bottom = [];
    // upper torso
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            ul_t_height, 
            ul_t_radius, 
            new BABYLON.Vector3(0,ul_t_height/2 + mid_t_height/2,0), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(1, 0, 0),
            [...head_connection, ...left_arm_torso_connection, ...right_arm_torso_connection],
            [],
            upper_torso_bottom)); 
    
    
    const left_leg_upper = [];
    const left_leg_lower = [];
    // left leg
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            leg_len, 
            leg_rad, 
            new BABYLON.Vector3(-leg_x_pos/2, -mid_t_height/2-ul_t_height-leg_len/2), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0,1,1),
            [],
            left_leg_upper,
            left_leg_lower));
    const right_leg_upper = [];
    const right_leg_lower = [];
    // right leg
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            leg_len, 
            leg_rad, 
            new BABYLON.Vector3(leg_x_pos/2, -mid_t_height/2-ul_t_height-leg_len/2), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0,1,1),
            [],
            right_leg_upper,
            right_leg_lower));
    
    // left leg base
    groups.push(
        makeBoxGroup(
            foot_pts, 
            foot_w, 
            foot_h, 
            foot_d, 
            new BABYLON.Vector3(foot_x_pos-foot_w/2, -mid_t_height/2-ul_t_height-leg_len-foot_h/2, 0.5), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0.7, 0.7, 0.2),
            left_leg_lower));
    // right leg base
    groups.push(
        makeBoxGroup(
            foot_pts, 
            foot_w, 
            foot_h, 
            foot_d, 
            new BABYLON.Vector3(foot_x_pos+foot_w/2, -mid_t_height/2-ul_t_height-leg_len-foot_h/2, 0.5), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0.7, 0.7, 0.2),
            right_leg_lower));

    const lower_torso_top = [];
    // lower torso
    groups.push(
        makeCylinderGroup(
            15, 
            ul_t_height, 
            ul_t_radius, 
            new BABYLON.Vector3(0,-ul_t_height/2 - mid_t_height/2,0), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0, 0, 1),
            [...left_leg_upper, ...right_leg_upper],
            lower_torso_top,
            [])); 
  
    // middle torso
    groups.push(
        makeCylinderGroup(
            cyl_sub, 
            mid_t_height, 
            mid_t_radius, 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Vector3(0,0,0), 
            new BABYLON.Color3(0, 1, 0),
            [...upper_torso_bottom, ...lower_torso_top],
            [],
            [])); 
    
    // left hand
    groups.push(
        makeBoxGroup(
            hand_pts, 
            hand_w, 
            hand_h, 
            hand_d, 
            new BABYLON.Vector3(arm_len*1.7, ul_t_height + mid_t_height/2, 1.5 ), 
            new BABYLON.Vector3(Math.PI/4,-Math.PI/6,Math.PI/2), 
            new BABYLON.Color3(0.2, 0.7, 0.2),
            left_arm_hand_connection));
            
    // right hand
    groups.push(
        makeBoxGroup(
            hand_pts, 
            hand_w, 
            hand_h, 
            hand_d, 
            new BABYLON.Vector3(-arm_len*1.7, ul_t_height + mid_t_height/2, 1.5 ), 
            new BABYLON.Vector3(Math.PI/4,Math.PI/6,-Math.PI/2), 
            new BABYLON.Color3(0.2, 0.7, 0.2),
            right_arm_hand_connection));
    
    return groups;
}


function buildRandomExampleShape() {
    let selectedMesh = null;
    const selectedMeshIdx = Math.floor(Math.random()*3);
    console.log('selected mesh')
    if (selectedMeshIdx === 0) {
        selectedMesh = makeBoxGroup(3, 10,10,10, new BABYLON.Vector3(0,0,0),new BABYLON.Vector3(0,0,0), null, []);
    } else if (selectedMeshIdx === 1) {
        selectedMesh =  makeCylinderGroup(18, 10, 10, new BABYLON.Vector3(0,0,0), new BABYLON.Vector3(0,0,0), null, [], [], []);
    } else if (selectedMeshIdx === 2) {
        selectedMesh = makeSphereGroup(44, 6, new BABYLON.Vector3(0,0,0), null, 0, []);
    }
    console.log('selectedMesh', selectedMesh);
    return [selectedMesh];
}

function setupSidebarResize() {
    const sidebar = document.getElementById("sidebar");
    const resizeHandle = document.getElementById("resizeHandle");
    const canvasContainer = document.getElementById("canvasContainer");
    
    if (!sidebar || !resizeHandle || !canvasContainer) {
        return;
    }
    
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    // Load saved width from localStorage
    const savedWidth = localStorage.getItem('sidebarWidth');
    if (savedWidth) {
        const width = parseInt(savedWidth, 10);
        if (width >= 280 && width <= 600) {
            sidebar.style.width = width + 'px';
        }
    }
    
    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = sidebar.offsetWidth;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        resizeHandle.style.background = 'var(--accent-primary)';
        e.preventDefault();
        e.stopPropagation();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const diff = e.clientX - startX;
        let newWidth = startWidth + diff;
        
        // Enforce min and max width
        const minWidth = 280;
        const maxWidth = 600;
        newWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
        
        sidebar.style.width = newWidth + 'px';
        
        // Trigger canvas resize if needed
        if (window.engine) {
            window.engine.resize();
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            resizeHandle.style.background = '';
            
            // Save width to localStorage
            localStorage.setItem('sidebarWidth', sidebar.offsetWidth.toString());
        }
    });
    
    // Add hover effect for better visibility and cursor feedback
    resizeHandle.addEventListener('mouseenter', () => {
        if (!isResizing) {
            document.body.style.cursor = 'col-resize';
            resizeHandle.style.background = 'rgba(99, 102, 241, 0.3)';
        }
    });
    
    resizeHandle.addEventListener('mouseleave', () => {
        if (!isResizing) {
            document.body.style.cursor = '';
            resizeHandle.style.background = '';
        }
    });
    
    // Prevent text selection while resizing
    resizeHandle.addEventListener('selectstart', (e) => {
        e.preventDefault();
    });
    
    // Make sure the resize handle is always on top and clickable
    resizeHandle.style.pointerEvents = 'auto';
    resizeHandle.style.zIndex = '10000';
}

async function main() {
    // Setup sidebar resize functionality
    setupSidebarResize();
    
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.Engine(canvas, true);
    window.engine = engine; // Store globally for resize access
    
    let scene = createScene(engine, canvas); //Call the createScene function
    showWorldAxis(1,scene);

    // Each entry is a group of points that represents a separate group on the input
    let groups = [];

    const randomExampleButton = document.getElementById("rndExample");
    if (randomExampleButton) {
    randomExampleButton.addEventListener("click", function() {
            updateStatus("Generating random example...", 'info');
            groups = clearScene(scene);
            showWorldAxis(1, scene);
            setTimeout(() => {
                groups = buildRandomExampleShape();
                updateStatus("✓ Random example loaded. Click 'Run Algorithm' to compute.", 'success');
            }, 100);
        });
    }

    // Start with theme groups by default
    groups = buildThemeGroups();
    updateStatus("✓ Ready. Theme example loaded. Click 'Run Algorithm' to see the convex hull, or load your own dataset.", 'success');
    
    // File input handler
    const fileInput = document.getElementById("fileInput");
    const loadFileButton = document.getElementById("loadFile");
    
    const loadFileHandler = async function() {
        const file = fileInput.files[0];
        if (!file) {
            updateStatus("⚠ Please select a file first", 'warning');
            return;
        }
        
        updateStatus(`Loading ${file.name}...`, 'info');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const text = e.target.result;
                const points = parsePointsFromText(text);
                
                if (points.length < 4) {
                    updateStatus(`✗ Error: Need at least 4 points, got ${points.length}`, 'error');
                    return;
                }
                
                // Clear everything first
                groups = clearScene(scene);
                showWorldAxis(1, scene);
                
                // Small delay to ensure scene is cleared
                setTimeout(() => {
                    const group = createGroupFromPoints(points, new BABYLON.Color3(0.8, 0.8, 1.0), scene);
                    if (group) {
                        groups = [group];
                        updateStatus(`✓ Loaded ${points.length} points from ${file.name}. Click 'Run Algorithm' to compute.`, 'success');
                    } else {
                        updateStatus(`✗ Failed to create point group`, 'error');
                    }
                }, 100);
            } catch (error) {
                updateStatus(`✗ Error loading file: ${error.message}`, 'error');
                console.error(error);
            }
        };
        reader.onerror = function() {
            updateStatus(`✗ Error reading file`, 'error');
        };
        reader.readAsText(file);
    };
    
    if (loadFileButton && fileInput) {
        loadFileButton.addEventListener("click", loadFileHandler);
        // Also auto-load when file is selected
        fileInput.addEventListener("change", loadFileHandler);
    }


    // Sample file buttons
    const loadSample1Btn = document.getElementById("loadSample1");
    if (loadSample1Btn) {
        loadSample1Btn.addEventListener("click", async function() {
            updateStatus("Loading Cube dataset...", 'info');
            const points = await loadSampleFile("sample_3d_points.txt");
            if (points) {
                groups = clearScene(scene);
                showWorldAxis(1, scene);
                setTimeout(() => {
                    const group = createGroupFromPoints(points, new BABYLON.Color3(1.0, 0.8, 0.8), scene);
                    if (group) {
                        groups = [group];
                        updateStatus(`✓ Loaded ${points.length} points (Cube). Click 'Run Algorithm' to compute.`, 'success');
                    }
                }, 100);
            }
        });
    }

    const loadSample2Btn = document.getElementById("loadSample2");
    if (loadSample2Btn) {
        loadSample2Btn.addEventListener("click", async function() {
            updateStatus("Loading Tetrahedron dataset...", 'info');
            const points = await loadSampleFile("test_points_2.txt");
            if (points) {
                groups = clearScene(scene);
                showWorldAxis(1, scene);
                setTimeout(() => {
                    const group = createGroupFromPoints(points, new BABYLON.Color3(0.8, 1.0, 0.8), scene);
                    if (group) {
                        groups = [group];
                        updateStatus(`✓ Loaded ${points.length} points (Tetrahedron). Click 'Run Algorithm' to compute.`, 'success');
                    }
                }, 100);
            }
        });
    }

    const loadSample3Btn = document.getElementById("loadSample3");
    if (loadSample3Btn) {
        loadSample3Btn.addEventListener("click", async function() {
            updateStatus("Loading Octahedron dataset...", 'info');
            const points = await loadSampleFile("test_points_3.txt");
            if (points) {
                groups = clearScene(scene);
                showWorldAxis(1, scene);
                setTimeout(() => {
                    const group = createGroupFromPoints(points, new BABYLON.Color3(0.8, 0.8, 1.0), scene);
                    if (group) {
                        groups = [group];
                        updateStatus(`✓ Loaded ${points.length} points (Octahedron). Click 'Run Algorithm' to compute.`, 'success');
                    }
                }, 100);
            }
        });
    }

    // Icosahedron dataset (12 vertices)
    const loadSample4Btn = document.getElementById("loadSample4");
    if (loadSample4Btn) {
        loadSample4Btn.addEventListener("click", async function() {
            updateStatus("Loading Icosahedron dataset...", 'info');
            // Generate Icosahedron points programmatically
            const phi = (1 + Math.sqrt(5)) / 2; // Golden ratio
            const points = [
                new BABYLON.Vector3(0, 1, phi),
                new BABYLON.Vector3(0, 1, -phi),
                new BABYLON.Vector3(0, -1, phi),
                new BABYLON.Vector3(0, -1, -phi),
                new BABYLON.Vector3(1, phi, 0),
                new BABYLON.Vector3(1, -phi, 0),
                new BABYLON.Vector3(-1, phi, 0),
                new BABYLON.Vector3(-1, -phi, 0),
                new BABYLON.Vector3(phi, 0, 1),
                new BABYLON.Vector3(phi, 0, -1),
                new BABYLON.Vector3(-phi, 0, 1),
                new BABYLON.Vector3(-phi, 0, -1)
            ];
            // Scale to reasonable size
            points.forEach(p => {
                p.scaleInPlace(3);
                p.id = ID_COUNTER++;
            });
            groups = clearScene(scene);
            showWorldAxis(1, scene);
            setTimeout(() => {
                const group = createGroupFromPoints(points, new BABYLON.Color3(1.0, 0.8, 1.0), scene);
                if (group) {
                    groups = [group];
                    updateStatus(`✓ Loaded ${points.length} points (Icosahedron). Click 'Run Algorithm' to compute.`, 'success');
                }
            }, 100);
        });
    }

    // Clear scene button
    const clearSceneBtn = document.getElementById("clearScene");
    if (clearSceneBtn) {
        clearSceneBtn.addEventListener("click", function() {
            updateStatus("Clearing scene...", 'info');
            groups = clearScene(scene);
            showWorldAxis(1, scene);
            hideAlgorithmStats();
            updateStatus("✓ Scene cleared. Load new points to continue.", 'success');
        });
    }

    // Theme button (if exists in HTML)
    const themeButton = document.getElementById("theme");
    if (themeButton) {
    themeButton.addEventListener("click", function() {
            updateStatus("Loading theme example...", 'info');
            groups = clearScene(scene);
            showWorldAxis(1, scene);
            setTimeout(() => {
        groups = buildThemeGroups();
                updateStatus("✓ Theme example loaded. Click 'Run Algorithm' to compute.", 'success');
            }, 100);
    });
    }
    
    // Register a render loop to repeatedly render the scene
    engine.runRenderLoop(function () {
        scene.render();
    });

    // Watch for browser/canvas resize events
    window.addEventListener("resize", function () {
        engine.resize();
    });


    // Animation speed control
    let currentAnimSpeed = 3.0;
    const animSpeedSlider = document.getElementById("animSpeedSlider");
    const animSpeedValue = document.getElementById("animSpeedValue");
    if (animSpeedSlider && animSpeedValue) {
        animSpeedSlider.addEventListener("input", function() {
            currentAnimSpeed = parseFloat(this.value);
            animSpeedValue.textContent = currentAnimSpeed.toFixed(1) + "x";
        });
    }

    // Get animation speed
    const getAnimSpeed = () => currentAnimSpeed;

    // Get selected algorithm
    const getSelectedAlgorithm = () => {
        const algorithmSelect = document.getElementById("algorithmSelect");
        return algorithmSelect ? algorithmSelect.value : 'quickhull';
    };

    // Stats tracking and display
    function displayAlgorithmStats(stats) {
        const statsDiv = document.getElementById("algorithmStats");
        const statsContent = document.getElementById("statsContent");
        
        if (!statsDiv || !statsContent) return;
        
        if (!stats || Object.keys(stats).length === 0) {
            statsDiv.style.display = 'none';
            return;
        }
        
        statsContent.innerHTML = '';
        
        // Format stats for display
        const formatTime = (ms) => {
            if (ms < 1) return `${(ms * 1000).toFixed(2)} μs`;
            if (ms < 1000) return `${ms.toFixed(2)} ms`;
            return `${(ms / 1000).toFixed(2)} s`;
        };
        
        const formatMemory = (bytes) => {
            if (bytes < 1024) return `${bytes} B`;
            if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
            return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
        };
        
        // Add each stat
        if (stats.time !== undefined) {
            const timeDiv = document.createElement('div');
            timeDiv.innerHTML = `<span style="color: var(--text-primary);">⏱️ Time:</span> <span>${formatTime(stats.time)}</span>`;
            statsContent.appendChild(timeDiv);
        }
        
        if (stats.vertices !== undefined) {
            const vertDiv = document.createElement('div');
            vertDiv.innerHTML = `<span style="color: var(--text-primary);">🔷 Vertices:</span> <span>${stats.vertices}</span>`;
            statsContent.appendChild(vertDiv);
        }
        
        if (stats.faces !== undefined) {
            const faceDiv = document.createElement('div');
            faceDiv.innerHTML = `<span style="color: var(--text-primary);">🔺 Faces:</span> <span>${stats.faces}</span>`;
            statsContent.appendChild(faceDiv);
        }
        
        if (stats.inputPoints !== undefined) {
            const inputDiv = document.createElement('div');
            inputDiv.innerHTML = `<span style="color: var(--text-primary);">📊 Input Points:</span> <span>${stats.inputPoints}</span>`;
            statsContent.appendChild(inputDiv);
        }
        
        if (stats.memory !== undefined && performance.memory) {
            const memDiv = document.createElement('div');
            memDiv.innerHTML = `<span style="color: var(--text-primary);">💾 Memory:</span> <span>${formatMemory(stats.memory)}</span>`;
            statsContent.appendChild(memDiv);
        }
        
        if (stats.iterations !== undefined) {
            const iterDiv = document.createElement('div');
            iterDiv.innerHTML = `<span style="color: var(--text-primary);">🔄 Iterations:</span> <span>${stats.iterations}</span>`;
            statsContent.appendChild(iterDiv);
        }
        
        statsDiv.style.display = 'block';
    }
    
    function hideAlgorithmStats() {
        const statsDiv = document.getElementById("algorithmStats");
        if (statsDiv) {
            statsDiv.style.display = 'none';
        }
    }

    const runQuickhullButton = document.getElementById("runQuickhull");
    if (runQuickhullButton) {
    runQuickhullButton.addEventListener("click", function() {
            if (groups.length === 0) {
                updateStatus("⚠ Error: No points loaded. Please load points first.", 'warning');
                return;
            }
            
            // Remove any existing hull meshes and separators
            scene.meshes.forEach(m => {
                if (m.name === "convex-hull" || m.name === "incremental-hull" || 
                    (m.name && m.name.includes("hull")) || 
                    (m.name && m.name.includes("jarvis")) ||
                    (m.name && m.name.includes("separator"))) {
                    m.dispose();
                }
            });
            
            try {
                const algorithm = getSelectedAlgorithm();
                const isCompareMode = algorithm === 'compare';
                const algorithmsToRun = isCompareMode ? ['quickhull', 'incremental', 'jarvis'] : [algorithm];
                
                let algorithmName = 'QuickHull';
                if (algorithm === 'jarvis') {
                    algorithmName = 'Jarvis March';
                } else if (algorithm === 'incremental') {
                    algorithmName = 'Incremental';
                } else if (isCompareMode) {
                    algorithmName = 'All 3 Algorithms';
                }
                
                updateStatus(`Computing ${algorithmName}...`, 'info');
                hideAlgorithmStats(); // Hide stats initially
                
                let totalPoints = 0;
                let computedGroups = 0;
                const allStats = [];
                
                // For compare mode, position hulls side-by-side with more spacing
                // Increased spacing to prevent overlap, especially for figures with extended arms
                const spacing = isCompareMode ? 50 : 0;
                
                groups.forEach(group => {
                    if (group.points && group.points.length >= 4) {
                        try {
                            totalPoints += group.points.length;
                            
                            algorithmsToRun.forEach((algo, algoIdx) => {
                                // Track timing and stats
                                const startTime = performance.now();
                                const startMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
                                
                                const hull = constructHull(group.points, algo);
                                
                                const endTime = performance.now();
                                const endMemory = performance.memory ? performance.memory.usedJSHeapSize : 0;
                                const computeTime = endTime - startTime;
                                const memoryUsed = endMemory - startMemory;
                                
                                if (!hull) {
                                    console.warn(`Failed to construct ${algo} hull for group with ${group.points.length} points`);
                                    return;
                                }
                                
                                // Store hull in appropriate property
                                if (isCompareMode) {
                                    if (!group.compareHulls) group.compareHulls = {};
                                    group.compareHulls[algo] = hull;
                                } else {
                                    group.hull = hull;
                                }
                                
                                if (typeof hull.buildRenderableMesh === 'function') {
                                    hull.buildRenderableMesh(scene, group.singleCol);
                                } else {
                                    console.warn(`Hull object doesn't have buildRenderableMesh method`);
                                    return;
                                }
                                
                                // Only set position/rotation if renderableMesh exists
                                if (hull.renderableMesh) {
                                    const basePos = group.renderPos || new BABYLON.Vector3(0, 0, 0);
                                    const offset = isCompareMode ? new BABYLON.Vector3((algoIdx - 1) * spacing, 0, 0) : new BABYLON.Vector3(0, 0, 0);
                                    hull.renderableMesh.position = basePos.add(offset);
                                    hull.renderableMesh.rotation = group.renderRot || new BABYLON.Vector3(0, 0, 0);
                                    
                                    // Ensure mesh is visible
                                    hull.renderableMesh.isVisible = true;
                                    hull.renderableMesh.setEnabled(true);
                                    
                                    // Log mesh details for debugging
                                    console.log(`Mesh created: ${hull.renderableMesh.name}`);
                                    console.log(`  Position:`, hull.renderableMesh.position);
                                    console.log(`  Vertices: ${hull.renderableMesh.getTotalVertices()}`);
                                    console.log(`  Faces: ${hull.faces ? hull.faces.length : 0}`);
                                    console.log(`  Has animation: ${!!hull.constructionAnimation}`);
                                    
                                    // Collect stats
                                    const stats = {
                                        algorithm: algo,
                                        time: computeTime,
                                        vertices: hull.renderableMesh.getTotalVertices(),
                                        faces: hull.faces ? hull.faces.length : 0,
                                        inputPoints: group.points.length,
                                        memory: memoryUsed > 0 ? memoryUsed : undefined
                                    };
                                    
                                    // Add iterations if available
                                    if (hull.iterations !== undefined) {
                                        stats.iterations = hull.iterations;
                                    }
                                    
                                    allStats.push(stats);
                                    
                                    // Check if mesh actually has geometry
                                    if (hull.renderableMesh.getTotalVertices() === 0) {
                                        console.error(`WARNING: Mesh has 0 vertices! This means no faces were computed.`);
                                        console.error(`  Input points: ${group.points.length}`);
                                        console.error(`  Algorithm faces: ${hull.faces ? hull.faces.length : 0}`);
                                    }
                                }
                            });
                            
                            computedGroups++;
                        } catch (err) {
                            console.error(`Error processing group with ${group.points.length} points:`, err);
                        }
                    }
                });
                
                // Add separator bars in compare mode
                if (isCompareMode && groups.length > 0) {
                    const group = groups[0];
                    const basePos = group.renderPos || new BABYLON.Vector3(0, 0, 0);
                    
                    // Create two separator planes between the three algorithms
                    for (let i = 0; i < 2; i++) {
                        const separator = BABYLON.MeshBuilder.CreatePlane("separator" + i, {
                            width: 0.1,
                            height: 20
                        }, scene);
                        
                        const xPos = basePos.x + ((i === 0 ? -0.5 : 0.5) * spacing);
                        separator.position = new BABYLON.Vector3(xPos, basePos.y, basePos.z);
                        separator.rotation.y = Math.PI / 2; // Rotate to face camera
                        
                        const separatorMat = new BABYLON.StandardMaterial("separatorMat" + i, scene);
                        separatorMat.diffuseColor = new BABYLON.Color3(0.3, 0.3, 0.3);
                        separatorMat.alpha = 0.3;
                        separatorMat.backFaceCulling = false;
                        separator.material = separatorMat;
                    }
                }
                
                updateStatus(`✓ ${algorithmName} computed: ${computedGroups} group(s), ${totalPoints} total points`, 'success');
                console.log(`${algorithmName} construction done`);
                
                // Display stats
                if (allStats.length > 0) {
                    if (isCompareMode) {
                        // Aggregate stats for compare mode
                        const aggregatedStats = {
                            time: allStats.reduce((sum, s) => sum + s.time, 0),
                            vertices: allStats.reduce((sum, s) => sum + s.vertices, 0),
                            faces: allStats.reduce((sum, s) => sum + s.faces, 0),
                            inputPoints: totalPoints,
                            memory: allStats.reduce((sum, s) => sum + (s.memory || 0), 0) || undefined
                        };
                        displayAlgorithmStats(aggregatedStats);
                    } else {
                        // Show stats for single algorithm (use first group's stats)
                        displayAlgorithmStats(allStats[0]);
                    }
                }
            } catch (error) {
                const algorithm = getSelectedAlgorithm();
                const algorithmName = algorithm === 'jarvis' ? 'Jarvis March' : 'QuickHull';
                updateStatus(`✗ Error computing ${algorithmName}: ${error.message}`, 'error');
                console.error(error);
            }
        });
    }

    const showQuickhullButton = document.getElementById("showQuickhull");
    if (showQuickhullButton) {
    showQuickhullButton.addEventListener("click", function() {
            const algorithm = getSelectedAlgorithm();
            const isCompareMode = algorithm === 'compare';
            
            if (groups.length === 0) {
                updateStatus("⚠ Error: Run algorithm first", 'warning');
                return;
            }
            
            // Check if hulls exist
            const hasHulls = isCompareMode ? 
                (groups[0].compareHulls && Object.keys(groups[0].compareHulls).length > 0) :
                groups[0].hull;
                
            if (!hasHulls) {
                updateStatus("⚠ Error: Run algorithm first", 'warning');
                return;
            }
            
            let animated = 0;
            let noAnimationCount = 0;
            
            groups.forEach((group, idx) => {
                if (isCompareMode && group.compareHulls) {
                    // Animate all three algorithms in compare mode
                    ['quickhull', 'incremental', 'jarvis'].forEach(algo => {
                        const hull = group.compareHulls[algo];
                        if (hull && hull.constructionAnimation) {
                            console.log(`Starting ${algo} animation for group ${idx}`);
                            hull.constructionAnimation.start(false, getAnimSpeed());
                            animated++;
                        } else if (hull) {
                            console.warn(`Group ${idx} ${algo} has no constructionAnimation property`);
                            noAnimationCount++;
                        }
                    });
                } else if (group.hull) {
                    // Single algorithm mode
                    if (group.hull.constructionAnimation) {
                        console.log(`Starting animation for group ${idx}`);
                        group.hull.constructionAnimation.start(false, getAnimSpeed());
                        animated++;
                    } else {
                        console.warn(`Group ${idx} has no constructionAnimation property`);
                        console.warn(`  Hull type: ${group.hull.constructor.name}`);
                        console.warn(`  Has renderableMesh: ${!!group.hull.renderableMesh}`);
                        console.warn(`  Has faces: ${!!group.hull.faces}, count: ${group.hull.faces ? group.hull.faces.length : 0}`);
                        noAnimationCount++;
                    }
                } else {
                    console.warn(`Group ${idx} has no hull`);
                }
            });
            
            if (animated > 0) {
                updateStatus(`✓ Playing animation for ${animated} algorithm(s)`, 'success');
            } else {
                updateStatus(`⚠ No animation available (${noAnimationCount} groups without animation)`, 'warning');
            }
        });
    }

}

main().then(() => {});