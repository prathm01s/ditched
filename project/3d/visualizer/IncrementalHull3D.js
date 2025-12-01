/**
 * 3D Convex Hull - Incremental Algorithm Implementation
 * Ported from Python to JavaScript for Babylon.js visualization
 * 
 * Algorithm: Start with an initial tetrahedron, then add remaining points one by one.
 * Time Complexity: O(n²) average case, O(n²) worst case
 * Version: 2024-11-19-FINAL - Complete working implementation
 */

import {Face, FaceTypes} from './Face.js';
import {signedDistanceToPlane} from './utils.js';
import { CONSTS } from './consts.js';

class IncrementalHull3D {
    constructor() {
        this.vertexList = []; // List of BABYLON.Vector3 objects
        this.faces = []; // List of Face objects
        this.renderableMesh = new BABYLON.Mesh("incremental-hull");
        this.epsilon = 1e-9;
        this.totalSteps = 0;
        this.usedInTetrahedron = new Set(); // Track which points were used in initial tetrahedron
    }

    /**
     * Build the convex hull using the incremental algorithm
     * @param {BABYLON.Vector3[]} inputPoints - Array of 3D points
     */
    build(inputPoints) {
        this.vertexList = [...inputPoints];
        
        if (this.vertexList.length < 4) {
            console.error(`Error: Need at least 4 points, got ${this.vertexList.length}`);
            return false;
        }

        let step = 0;
        
        // Step 1: Initialize with a tetrahedron
        if (!this._initializeTetrahedron(step)) {
            console.error("Error: Could not find 4 non-coplanar points");
            return false;
        }
        step += 1;

        // Step 2: Add remaining points one by one (skip points used in tetrahedron)
        for (let i = 0; i < this.vertexList.length; i++) {
            if (!this.usedInTetrahedron.has(i)) {
                this._addPoint(i, step);
                step += 1;
            }
        }

        this.totalSteps = step;
        
        // Debug: Count active faces
        const activeFaceCount = this.faces.filter(f => f.mark !== FaceTypes.DELETED).length;
        console.log(`Incremental hull construction done. Total faces: ${this.faces.length}, Active: ${activeFaceCount}`);
        return true;
    }

    /**
     * Initialize the convex hull with the first 4 non-coplanar points
     * @param {number} step - Current step number for animation
     * @returns {boolean} True if successful
     */
    _initializeTetrahedron(step) {
        const maxSearch = Math.min(10, this.vertexList.length);
        
        // Find 4 non-coplanar points
        for (let i = 0; i < maxSearch; i++) {
            for (let j = i + 1; j < maxSearch; j++) {
                for (let k = j + 1; k < maxSearch; k++) {
                    for (let l = k + 1; l < maxSearch; l++) {
                        if (this._areNonCoplanar(i, j, k, l)) {
                            // Create 4 faces of the tetrahedron
                            // Ensure correct orientation (outward facing)
                            // We need to check if the 4th point is "behind" the face
                            
                            const p0 = this.vertexList[i];
                            const p1 = this.vertexList[j];
                            const p2 = this.vertexList[k];
                            const p3 = this.vertexList[l];
                            
                            // Compute centroid of the tetrahedron
                            const center = p0.add(p1).add(p2).add(p3).scale(0.25);
                            
                            // Helper to create face with correct orientation
                            // The normal should point AWAY from the center (outward)
                            const createOrientedFace = (a, b, c) => {
                                const f = new Face(step);
                                f.buildFromPoints(a, b, c);
                                
                                // Vector from face to center
                                const facePoint = a; // Any point on the face
                                const toCenter = center.subtract(facePoint);
                                
                                // If normal points toward center (dot > 0), flip it
                                if (BABYLON.Vector3.Dot(f.normal, toCenter) > 0) {
                                    f.buildFromPoints(a, c, b);
                                }
                                return f;
                            };
                            
                            // Create the 4 faces of the tetrahedron
                            // Each face uses 3 of the 4 points
                            const face1 = createOrientedFace(p0, p1, p2); // opposite to p3
                            const face2 = createOrientedFace(p0, p1, p3); // opposite to p2
                            const face3 = createOrientedFace(p0, p2, p3); // opposite to p1
                            const face4 = createOrientedFace(p1, p2, p3); // opposite to p0
                            
                            this.faces = [face1, face2, face3, face4];
                            
                            // Track which points were used in the tetrahedron
                            this.usedInTetrahedron.add(i);
                            this.usedInTetrahedron.add(j);
                            this.usedInTetrahedron.add(k);
                            this.usedInTetrahedron.add(l);
                            
                            console.log(`Initial tetrahedron: using points ${i}, ${j}, ${k}, ${l}`);
                            console.log(`  Face 1 normal: ${face1.normal.toString()}`);
                            console.log(`  Face 1 points: ${face1.points.length}`);
                            
                            return true;
                        }
                    }
                }
            }
        }
        
        return false;
    }

    /**
     * Check if 4 points are non-coplanar using the volume test
     * @param {number} i - Index of first point
     * @param {number} j - Index of second point
     * @param {number} k - Index of third point
     * @param {number} l - Index of fourth point
     * @returns {boolean} True if non-coplanar
     */
    _areNonCoplanar(i, j, k, l) {
        const v0 = this.vertexList[i];
        const v1 = this.vertexList[j];
        const v2 = this.vertexList[k];
        const v3 = this.vertexList[l];
        
        // Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
        const edge1 = v1.subtract(v0);
        const edge2 = v2.subtract(v0);
        const edge3 = v3.subtract(v0);
        
        // Compute scalar triple product (cross product then dot product)
        const cross = BABYLON.Vector3.Cross(edge1, edge2);
        const volume = Math.abs(BABYLON.Vector3.Dot(cross, edge3));
        
        return volume > this.epsilon;
    }

    /**
     * Add a single point to the existing convex hull
     * @param {number} pointIdx - Index of the point to add
     * @param {number} step - Current step number for animation
     */
    _addPoint(pointIdx, step) {
        const point = this.vertexList[pointIdx];
        
        // Get only active (non-deleted) faces for efficiency
        const activeFaces = [];
        const activeFaceIndices = [];
        for (let faceIdx = 0; faceIdx < this.faces.length; faceIdx++) {
            if (this.faces[faceIdx].mark !== FaceTypes.DELETED) {
                activeFaces.push(this.faces[faceIdx]);
                activeFaceIndices.push(faceIdx);
            }
        }
        
        // Step 1: Determine visibility of each face
        const visibleFaces = [];
        for (let i = 0; i < activeFaces.length; i++) {
            const face = activeFaces[i];
            
            // Compute signed distance from point to face plane
            // Use the face's normal if available, otherwise compute it
            let distance;
            if (face.normal && face.normal.lengthSquared() > 0) {
                // Use face normal for more accurate computation
                const v0 = face.points[0];
                const toPoint = point.subtract(v0);
                distance = BABYLON.Vector3.Dot(face.normal, toPoint);
            } else {
                // Fallback to signedDistanceToPlane
                distance = signedDistanceToPlane(
                    face.points[0],
                    face.points[1],
                    face.points[2],
                    point
                );
            }
            
            if (distance > this.epsilon) {
                // Point is on the positive side of the plane (visible)
                visibleFaces.push(activeFaceIndices[i]);
            }
        }
        
        // If point is inside the hull, don't add it
        if (visibleFaces.length === 0) {
            console.log(`Point ${pointIdx} is inside hull, skipping`);
            return;
        }
        
        console.log(`Point ${pointIdx} sees ${visibleFaces.length} faces out of ${activeFaces.length} active`);
        
        // Special case: if point sees ALL active faces, it might be on the wrong side
        // This shouldn't happen with correct orientation, so skip it
        if (visibleFaces.length === activeFaces.length) {
            console.warn(`Point ${pointIdx} sees ALL ${activeFaces.length} faces - likely orientation error. Skipping.`);
            return;
        }
        
        // Step 2: Find horizon edges (only check active faces)
        const horizonEdges = this._findHorizonEdges(visibleFaces, activeFaces, activeFaceIndices);
        
        if (horizonEdges.length === 0) {
            console.warn(`Point ${pointIdx} sees ${visibleFaces.length} faces but no horizon found. Skipping to preserve hull.`);
            return;
        }
        
        console.log(`Found ${horizonEdges.length} horizon edges`);
        
        // Step 3: Mark visible faces as deleted
        for (let faceIdx of visibleFaces) {
            this.faces[faceIdx].markAsDeleted(step);
        }
        
        // Step 4: Add new faces connecting point to horizon
        for (let edge of horizonEdges) {
            const newFace = new Face(step);
            newFace.buildFromPoints(
                edge[0],  // edge[0] and edge[1] are already Vector3 objects
                edge[1],
                point
            );
            this.faces.push(newFace);
        }
    }

    /**
     * Find the horizon edges from visible faces
     * An edge is a horizon edge if exactly one of its two adjacent faces is visible
     * @param {number[]} visibleFaceIndices - Indices of visible faces
     * @param {Face[]} activeFaces - Array of active (non-deleted) faces
     * @param {number[]} activeFaceIndices - Original indices of active faces
     * @returns {Array<[BABYLON.Vector3, BABYLON.Vector3]>} List of horizon edges as [v0, v1] pairs
     */
    _findHorizonEdges(visibleFaceIndices, activeFaces, activeFaceIndices) {
        const visibleSet = new Set(visibleFaceIndices);
        const edgeCount = new Map(); // Count how many visible faces share each edge
        const edgeToNonVisibleFace = new Map(); // Map edge to non-visible face edge
        
        // Helper function to normalize an edge (order vertices consistently)
        // Creates a consistent key for edge matching
        const normalizeEdge = (v1, v2) => {
            // Create key using rounded coordinates for robust matching
            const k1 = `${Math.round(v1.x * 1e6)},${Math.round(v1.y * 1e6)},${Math.round(v1.z * 1e6)}`;
            const k2 = `${Math.round(v2.x * 1e6)},${Math.round(v2.y * 1e6)},${Math.round(v2.z * 1e6)}`;
            
            // Order keys consistently (lexicographically)
            const key = k1 < k2 ? `${k1}|${k2}` : `${k2}|${k1}`;
            
            return {key: key, original: [v1, v2]};
        };
        
        // First pass: collect edges from visible faces
        for (let i = 0; i < activeFaces.length; i++) {
            const faceIdx = activeFaceIndices[i];
            if (!visibleSet.has(faceIdx)) {
                continue; // Skip non-visible faces in first pass
            }
            
            const face = activeFaces[i];
            const v = face.points;
            
            const edges = [
                [v[0], v[1]],
                [v[1], v[2]],
                [v[2], v[0]]
            ];
            
            for (let edge of edges) {
                const normEdge = normalizeEdge(edge[0], edge[1]);
                edgeCount.set(normEdge.key, (edgeCount.get(normEdge.key) || 0) + 1);
            }
        }
        
        // Second pass: find edges from non-visible faces that are shared with visible faces
        for (let i = 0; i < activeFaces.length; i++) {
            const faceIdx = activeFaceIndices[i];
            if (visibleSet.has(faceIdx)) {
                continue; // Skip visible faces in second pass
            }
            
            const face = activeFaces[i];
            const v = face.points;
            
            const edges = [
                [v[0], v[1]],
                [v[1], v[2]],
                [v[2], v[0]]
            ];
            
            for (let edge of edges) {
                const normEdge = normalizeEdge(edge[0], edge[1]);
                if (edgeCount.has(normEdge.key)) {
                    // This edge is shared between visible and non-visible faces - it's a horizon edge
                    // Store the original edge direction from the non-visible face
                    edgeToNonVisibleFace.set(normEdge.key, edge);
                }
            }
        }
        
        // Collect horizon edges with correct orientation
        const horizon = [];
        for (let [edgeKey, originalEdge] of edgeToNonVisibleFace) {
            // Return edge in correct orientation (reversed for outward normal)
            horizon.push([originalEdge[1], originalEdge[0]]);
        }
        
        return horizon;
    }

    /**
     * Build the renderable mesh for Babylon.js
     * Similar to Quickhull3D's buildRenderableMesh
     * @param {BABYLON.Scene} scene - Babylon.js scene
     * @param {BABYLON.Color3} singleColor - Optional single color for all faces
     */
    buildRenderableMesh(scene, singleColor = null) {
        const vertexData = new BABYLON.VertexData();

        // Iterate over faces, adding vertices and building indices
        const vertices = [];
        const faces = [];
        const normals = [];
        const lifetime = [];
        
        // Only include faces that are not deleted - filter once for efficiency
        const activeFaces = [];
        for (let f of this.faces) {
            if (f.mark !== FaceTypes.DELETED) {
                activeFaces.push(f);
            }
        }
        
        console.log(`Building renderable mesh: ${activeFaces.length} active faces out of ${this.faces.length} total`);
        
        for (let f of activeFaces) {
            // Add vertices for this face - same order as QuickHull
            for (let v of f.points) {
                vertices.push(v.x, v.y, v.z);
            }
            
            const vlen = vertices.length / 3;
            // Indices in reverse order for correct winding (same as QuickHull)
            faces.push(vlen - 1, vlen - 2, vlen - 3);
            
            // Add normals (same for all three vertices of the face)
            const normal = f.normal;
            for (let i = 0; i < 3; i++) {
                normals.push(normal.x, normal.y, normal.z);
            }
            
            lifetime.push({
                createdAt: f.createdAt || 0,
                deletedAt: f.deletedAt
            });
        }

        const materials = [];
        const multimaterial = new BABYLON.MultiMaterial("incremental-hull", scene);
        
        const animations = new BABYLON.AnimationGroup("incremental-hull-anim");

        const START_OPACITY = 0.0;
        const END_OPACITY = CONSTS.HULL_OPACITY;

        const colArr = new Array(this.totalSteps);

        // Create materials and animations for each face
        for (let i = 0; i < lifetime.length; i++) {
            const {createdAt, deletedAt} = lifetime[i];

            const nm = new BABYLON.StandardMaterial("face" + i);
            nm.backFaceCulling = false;

            if (singleColor === null) {
                let col = colArr[createdAt];
                if (col === null || col === undefined) {
                    col = new BABYLON.Color3(Math.random(), Math.random(), Math.random());
                }
                colArr[createdAt] = col;
                nm.diffuseColor = col;
            } else {
                nm.diffuseColor = singleColor;
            }
        
            materials.push(nm);
            multimaterial.subMaterials.push(nm);

            const animation = new BABYLON.Animation("face" + i, "alpha", CONSTS.FPS, BABYLON.Animation.ANIMATIONTYPE_FLOAT);
            const keys = [];
            keys.push(
                {frame: 0, value: 0},
                {frame: createdAt * CONSTS.FPS, value: START_OPACITY},
                {frame: (createdAt + 1) * CONSTS.FPS, value: END_OPACITY}
            );

            if (deletedAt) {
                keys.push(
                    {frame: deletedAt * CONSTS.FPS, value: END_OPACITY},
                    {frame: (deletedAt + 1) * CONSTS.FPS, value: START_OPACITY}
                );
            }

            animation.setKeys(keys);
            animations.addTargetedAnimation(animation, nm);
        }

        animations.normalize();

        vertexData.positions = vertices;
        vertexData.indices = faces;
        vertexData.normals = normals;

        vertexData.applyToMesh(this.renderableMesh);

        // Create a submesh for every face
        this.renderableMesh.subMeshes = [];
        for (let i = 0; i < lifetime.length; i++) {
            new BABYLON.SubMesh(i, 0, vertices.length / 3, i * 3, 3, this.renderableMesh);
        }

        this.renderableMesh.material = multimaterial;
        this.constructionAnimation = animations;
        
        console.log('Incremental hull renderable mesh built');
    }
}

export {IncrementalHull3D};

