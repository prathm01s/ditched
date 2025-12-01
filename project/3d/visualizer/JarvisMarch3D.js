// JarvisMarch3D.js  (Patched — Babylon v8 safe, dedupe, clean & validate faces, single material)
// Version: 2024-11-19-FINAL - Complete working implementation with dihedral angle-based gift-wrapping
import { Face } from './Face.js';
import { CONSTS } from './consts.js';

export class JarvisMarch3D {
    constructor(vertexList, step) {
        this.vertexList = vertexList || []; // array of BABYLON.Vector3
        this.step = step;
        this.faces = [];
        this.epsilon = 1e-9;
        this.renderableMesh = null;
        this.faceCreationOrder = []; // Track order of face creation for animation
        this.constructionAnimation = null;
    }

    // ---------- Vector helpers ----------
    vecKey(v) {
        return `${v.x.toFixed(6)},${v.y.toFixed(6)},${v.z.toFixed(6)}`;
    }

    vecEquals(a, b, eps = this.epsilon) {
        return Math.abs(a.x - b.x) <= eps &&
               Math.abs(a.y - b.y) <= eps &&
               Math.abs(a.z - b.z) <= eps;
    }

    vecAdd(a, b) { return new BABYLON.Vector3(a.x + b.x, a.y + b.y, a.z + b.z); }
    vecSubtract(a, b) { return new BABYLON.Vector3(a.x - b.x, a.y - b.y, a.z - b.z); }
    vecScale(a, s) { return new BABYLON.Vector3(a.x * s, a.y * s, a.z * s); }
    vecCross(a, b) {
        if (BABYLON.Vector3 && typeof BABYLON.Vector3.Cross === 'function') return BABYLON.Vector3.Cross(a, b);
        return new BABYLON.Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    vecDot(a, b) { if (BABYLON.Vector3 && typeof BABYLON.Vector3.Dot === 'function') return BABYLON.Vector3.Dot(a, b); return a.x*b.x + a.y*b.y + a.z*b.z; }
    vecLengthSq(a) { return a.x*a.x + a.y*a.y + a.z*a.z; }
    vecNormalize(a) { const len = Math.sqrt(this.vecLengthSq(a)); if (len <= this.epsilon) return new BABYLON.Vector3(0,0,0); return this.vecScale(a, 1.0/len); }

    // ---------- Geometry helpers ----------
    signedVolume(a, b, c, d) {
        const ab = this.vecSubtract(b, a);
        const ac = this.vecSubtract(c, a);
        const ad = this.vecSubtract(d, a);
        const crossABAC = this.vecCross(ab, ac);
        return this.vecDot(crossABAC, ad);
    }

    findBottommostPoint() {
        let bottomIdx = 0;
        const points = this.vertexList;
        for (let i = 1; i < points.length; ++i) {
            const p1 = points[bottomIdx];
            const p2 = points[i];
            if (p2.z < p1.z ||
                (Math.abs(p2.z - p1.z) < this.epsilon && p2.y < p1.y) ||
                (Math.abs(p2.z - p1.z) < this.epsilon &&
                 Math.abs(p2.y - p1.y) < this.epsilon &&
                 p2.x < p1.x)) {
                bottomIdx = i;
            }
        }
        return bottomIdx;
    }

    isValidHullFace(p1, p2, p3) {
        const hullVertices = new Set();
        for (const face of this.faces) {
            if (face.points && face.points.length >= 3) {
                for (const pt of face.points) hullVertices.add(this.vecKey(pt));
            }
        }
        hullVertices.add(this.vecKey(p1)); hullVertices.add(this.vecKey(p2)); hullVertices.add(this.vecKey(p3));

        for (let i = 0; i < this.vertexList.length; ++i) {
            const p = this.vertexList[i];
            const key = this.vecKey(p);
            if (hullVertices.has(key)) continue;
            const vol = this.signedVolume(p1, p2, p3, p);
            if (Math.abs(vol) <= this.epsilon) continue;
            if (vol < -this.epsilon) return false;
        }
        return true;
    }

    findNextFacePoint(edgeP1, edgeP2, currentFaceNormal) {
        let bestIdx = -1;
        let minAngle = Infinity;  // We want the MINIMUM dihedral angle (rightmost/tightest turn)

        const edgeVec = this.vecSubtract(edgeP2, edgeP1);
        const edgeNorm = this.vecNormalize(edgeVec);

        // Reference perpendicular to edge for angle measurement
        let refPerp = null;
        if (currentFaceNormal && this.vecLengthSq(currentFaceNormal) > this.epsilon) {
            // Project current normal onto plane perpendicular to edge
            const normalNorm = this.vecNormalize(currentFaceNormal);
            const dotEdge = this.vecDot(normalNorm, edgeNorm);
            refPerp = this.vecSubtract(normalNorm, this.vecScale(edgeNorm, dotEdge));
            refPerp = this.vecNormalize(refPerp);
        }

        for (let i = 0; i < this.vertexList.length; ++i) {
            const p = this.vertexList[i];
            if (this.vecEquals(p, edgeP1) || this.vecEquals(p, edgeP2)) continue;

            // skip if p is already a vertex of a face containing this edge
            let skip = false;
            for (const face of this.faces) {
                if (!face.points || face.points.length < 3) continue;
                const fp = face.points;
                const containsEdge =
                    (this.vecEquals(fp[0], edgeP1) && this.vecEquals(fp[1], edgeP2)) ||
                    (this.vecEquals(fp[1], edgeP1) && this.vecEquals(fp[2], edgeP2)) ||
                    (this.vecEquals(fp[2], edgeP1) && this.vecEquals(fp[0], edgeP2)) ||
                    (this.vecEquals(fp[0], edgeP2) && this.vecEquals(fp[1], edgeP1)) ||
                    (this.vecEquals(fp[1], edgeP2) && this.vecEquals(fp[2], edgeP1)) ||
                    (this.vecEquals(fp[2], edgeP2) && this.vecEquals(fp[0], edgeP1));
                if (containsEdge && (this.vecEquals(fp[0], p) || this.vecEquals(fp[1], p) || this.vecEquals(fp[2], p))) { skip = true; break; }
            }
            if (skip) continue;

            // Compute perpendicular from point to edge
            const toPoint = this.vecSubtract(p, edgeP1);
            const projOnEdge = this.vecDot(toPoint, edgeNorm);
            const perpToEdge = this.vecSubtract(toPoint, this.vecScale(edgeNorm, projOnEdge));
            const perpNorm = this.vecNormalize(perpToEdge);
            
            if (this.vecLengthSq(perpToEdge) < this.epsilon) continue; // Point is on the edge
            
            // Compute angle
            let angle;
            if (refPerp && this.vecLengthSq(refPerp) > this.epsilon) {
                // Compute signed angle in the plane perpendicular to edge
                const cosAngle = this.vecDot(refPerp, perpNorm);
                const sinAngle = this.vecDot(edgeNorm, this.vecCross(refPerp, perpNorm));
                angle = Math.atan2(sinAngle, cosAngle);
                
                // Normalize to [0, 2π]
                if (angle < 0) angle += 2 * Math.PI;
            } else {
                // No reference: pick any point (use smallest index as tiebreaker)
                angle = i;
            }
            
            if (angle < minAngle) {
                minAngle = angle;
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    edgeKey(v1, v2) { const k1 = this.vecKey(v1); const k2 = this.vecKey(v2); return [k1,k2].sort().join('|'); }
    faceKey(v1, v2, v3) { const k1 = this.vecKey(v1); const k2 = this.vecKey(v2); const k3 = this.vecKey(v3); return [k1,k2,k3].sort().join('|'); }

    countFacesWithEdge(v1, v2) {
        let count = 0;
        for (const face of this.faces) {
            if (!face.points || face.points.length < 3) continue;
            const p = face.points;
            const hasEdge =
                (this.vecEquals(p[0], v1) && this.vecEquals(p[1], v2)) ||
                (this.vecEquals(p[1], v1) && this.vecEquals(p[2], v2)) ||
                (this.vecEquals(p[2], v1) && this.vecEquals(p[0], v2)) ||
                (this.vecEquals(p[0], v2) && this.vecEquals(p[1], v1)) ||
                (this.vecEquals(p[1], v2) && this.vecEquals(p[2], v1)) ||
                (this.vecEquals(p[2], v2) && this.vecEquals(p[0], v1));
            if (hasEdge) count++;
        }
        return count;
    }

    // ---------- New: clean degenerate/duplicate faces ----------
    cleanFaces() {
        const kept = [];
        const seen = new Set();
        const areaThreshold = Math.max(this.epsilon * 100, 1e-12);

        for (const face of this.faces) {
            if (!face || !face.points || face.points.length < 3) continue;
            const v1 = face.points[0], v2 = face.points[1], v3 = face.points[2];

            // degenerate if vertices coincide
            if (this.vecEquals(v1, v2) || this.vecEquals(v2, v3) || this.vecEquals(v3, v1)) continue;

            // triangle area
            const cross = this.vecCross(this.vecSubtract(v2, v1), this.vecSubtract(v3, v1));
            const area = Math.sqrt(this.vecDot(cross, cross)) * 0.5;
            if (area <= areaThreshold) continue;

            const key = [this.vecKey(v1), this.vecKey(v2), this.vecKey(v3)].sort().join('|');
            if (seen.has(key)) continue;
            seen.add(key);
            kept.push(face);
        }
        const removed = this.faces.length - kept.length;
        this.faces = kept;
        console.log(`Jarvis March: cleanFaces() removed ${removed} degenerate/duplicate faces (${kept.length} kept)`);
    }
    
    // ---------- Check for nearly coplanar faces (z-fighting prevention) ----------
    removeCoplanarFaces() {
        if (this.faces.length < 2) return;
        
        const kept = [];
        const removed = [];
        const angleThreshold = 0.9999; // cos(0.8 degrees) - very strict, only truly parallel
        
        // Helper to check if two faces have very similar (but not identical) vertices
        const haveSimilarVertices = (f1pts, f2pts, tolerance) => {
            let similarCount = 0;
            for (let v1 of f1pts) {
                for (let v2 of f2pts) {
                    const dist = Math.sqrt(this.vecLengthSq(this.vecSubtract(v1, v2)));
                    if (dist < tolerance) {
                        similarCount++;
                        break;
                    }
                }
            }
            return similarCount >= 2; // At least 2 vertices are very close
        };
        
        for (let i = 0; i < this.faces.length; i++) {
            const face = this.faces[i];
            if (!face || !face.points || face.points.length < 3) continue;
            
            const v1 = face.points[0], v2 = face.points[1], v3 = face.points[2];
            const edge1 = this.vecSubtract(v2, v1);
            const edge2 = this.vecSubtract(v3, v1);
            const normal = this.vecNormalize(this.vecCross(edge1, edge2));
            const faceCenter = this.vecScale(this.vecAdd(this.vecAdd(v1, v2), v3), 1.0/3);
            
            // Compute face characteristic size for relative tolerance
            const faceSize = Math.sqrt(this.vecLengthSq(edge1) + this.vecLengthSq(edge2));
            
            let isOverlapping = false;
            
            // Check against all kept faces
            for (const keptFace of kept) {
                const kv1 = keptFace.points[0], kv2 = keptFace.points[1], kv3 = keptFace.points[2];
                const kedge1 = this.vecSubtract(kv2, kv1);
                const kedge2 = this.vecSubtract(kv3, kv1);
                const knormal = this.vecNormalize(this.vecCross(kedge1, kedge2));
                
                // Check if normals are nearly parallel (or anti-parallel)
                const dotProduct = Math.abs(this.vecDot(normal, knormal));
                if (dotProduct > angleThreshold) {
                    // Check if faces share EXACTLY the same vertices (adjacent triangles)
                    let exactSharedVertices = 0;
                    for (let fv of [v1, v2, v3]) {
                        for (let kv of [kv1, kv2, kv3]) {
                            if (this.vecEquals(fv, kv)) {
                                exactSharedVertices++;
                                break;
                            }
                        }
                    }
                    
                    // If they share 2+ EXACT vertices (an edge), they're adjacent - keep both
                    if (exactSharedVertices >= 2) {
                        continue;
                    }
                    
                    // Check if they have similar (but not identical) vertices
                    const keptCenter = this.vecScale(this.vecAdd(this.vecAdd(kv1, kv2), kv3), 1.0/3);
                    const centerDiff = this.vecSubtract(faceCenter, keptCenter);
                    const centerDistance = Math.sqrt(this.vecLengthSq(centerDiff));
                    
                    // Distance from face center to the other face's plane
                    const distToPlane = Math.abs(this.vecDot(this.vecSubtract(faceCenter, kv1), knormal));
                    
                    // More aggressive checks for overlapping faces:
                    // 1. Very close centers AND on same plane
                    // 2. OR similar vertices (near-duplicate face)
                    const relativeTol = faceSize * 0.01; // 1% of face size
                    const similarVerts = haveSimilarVertices([v1, v2, v3], [kv1, kv2, kv3], relativeTol);
                    
                    if ((centerDistance < relativeTol && distToPlane < relativeTol) || 
                        (similarVerts && distToPlane < relativeTol * 10)) {
                        isOverlapping = true;
                        removed.push(face);
                        break;
                    }
                }
            }
            
            if (!isOverlapping) {
                kept.push(face);
            }
        }
        
        if (removed.length > 0) {
            console.log(`Jarvis March: removeCoplanarFaces() removed ${removed.length} overlapping faces (z-fighting prevention)`);
            this.faces = kept;
        }
    }

    // ---------- New: Validate face orientation & remove internal faces ----------
    validateAndOrientFaces() {
        const finalFaces = [];
        let removedCount = 0;

        const totalPoints = this.vertexList.length;
        // Tolerance: allow up to N opposing points OR up to R fraction of total points
        // Increased tolerance: 5% or at least 2 points, to handle coplanar/near-coplanar cases
        const maxOpposingCount = Math.max(2, Math.floor(totalPoints * 0.05)); // default 5% or at least 2
        const maxOpposingRatio = 0.05; // 5%

        for (const face of this.faces) {
            if (!face || !face.points || face.points.length < 3) continue;
            let [v1, v2, v3] = face.points;

            let n = this.vecCross(this.vecSubtract(v2, v1), this.vecSubtract(v3, v1));
            n = this.vecNormalize(n);
            const d = -this.vecDot(n, v1);

            let pos = 0, neg = 0;
            const planeEpsilon = this.epsilon * 100; // Larger epsilon for coplanar detection
            for (const p of this.vertexList) {
                if (this.vecEquals(p, v1) || this.vecEquals(p, v2) || this.vecEquals(p, v3)) continue;
                const s = this.vecDot(n, p) + d;
                // Treat points very close to the plane as coplanar (not counted in pos/neg)
                if (Math.abs(s) <= planeEpsilon) continue;
                if (s > planeEpsilon) pos++;
                else if (s < -planeEpsilon) neg++;
            }

            // If points only on one side -> keep (or flip if majority negative)
            if (pos > 0 && neg === 0) {
                finalFaces.push(face);
                continue;
            }
            if (neg > 0 && pos === 0) {
                const flippedFace = new Face(this.step);
                flippedFace.buildFromPoints(v1, v3, v2);
                finalFaces.push(flippedFace);
                continue;
            }

            // Both sides present: allow small numerical disagreement
            const maxOpposing = Math.max(maxOpposingCount, Math.ceil(totalPoints * maxOpposingRatio));
            const smaller = Math.min(pos, neg);
            const larger = Math.max(pos, neg);
            const total = pos + neg;
            const ratio = total > 0 ? larger / total : 0;
            
            // If the smaller side is within tolerance, keep/flip based on majority
            if (smaller <= maxOpposing) {
                // pick the majority side (if pos > neg keep, if neg > pos flip)
                if (pos >= neg) finalFaces.push(face);
                else {
                    const flippedFace = new Face(this.step);
                    flippedFace.buildFromPoints(v1, v3, v2);
                    finalFaces.push(flippedFace);
                }
                continue;
            }
            
            // More lenient checks for faces with strong majorities
            // If ratio >= 0.7 (70%+ on one side), allow up to 30% on the smaller side
            if (ratio >= 0.7 && smaller <= Math.max(5, Math.floor(totalPoints * 0.30))) {
                // Strong majority: flip if needed, otherwise keep
                if (neg > pos) {
                    const flippedFace = new Face(this.step);
                    flippedFace.buildFromPoints(v1, v3, v2);
                    finalFaces.push(flippedFace);
                } else {
                    finalFaces.push(face);
                }
                continue;
            }
            
            // If ratio >= 0.6 (60%+ on one side), allow up to 20% on the smaller side
            if (ratio >= 0.6 && smaller <= Math.max(4, Math.floor(totalPoints * 0.20))) {
                // Moderate majority: flip if needed, otherwise keep
                if (neg > pos) {
                    const flippedFace = new Face(this.step);
                    flippedFace.buildFromPoints(v1, v3, v2);
                    finalFaces.push(flippedFace);
                } else {
                    finalFaces.push(face);
                }
                continue;
            }

            // Otherwise remove face as truly internal/conflicting
            // Only log if ratio is close to 0.5 (near 50/50 split), indicating a real conflict
            if (ratio < 0.65) {
                console.warn('Removing face due to conflict', {
                    faceKey: this.faceKey(v1, v2, v3),
                    pos, neg, totalPoints, ratio: ratio.toFixed(2)
                });
            }
            removedCount++;
        }

        this.faces = finalFaces;
        console.log(`Jarvis March: validateAndOrientFaces() removed ${removedCount} internal/conflicting faces, ${this.faces.length} remain`);
    }

    // ---------- Core algorithm ----------
    compute() {
        if (!this.vertexList || this.vertexList.length < 4) {
            console.warn("Jarvis March: Need at least 4 points");
            return;
        }

        this.faces = [];
        const processedFaces = new Set();
        const edgeQueue = [];

        // Step 1: bottommost
        const bottomIdx = this.findBottommostPoint();
        const bottomPoint = this.vertexList[bottomIdx];

        // Step 2: farthest from bottomPoint
        let maxDist = -1, secondIdx = -1;
        for (let i = 0; i < this.vertexList.length; ++i) {
            if (i === bottomIdx) continue;
            const v = this.vecSubtract(this.vertexList[i], bottomPoint);
            const dist = this.vecLengthSq(v);
            if (dist > maxDist) { maxDist = dist; secondIdx = i; }
        }
        if (secondIdx === -1) { console.warn("Jarvis March: All points identical"); return; }
        let secondPoint = this.vertexList[secondIdx];

        // Step 3: third point maximizing volume
        let maxVolume = -Infinity, thirdIdx = -1;
        const edgeVec = this.vecSubtract(secondPoint, bottomPoint);
        let edgePerp = this.vecCross(edgeVec, new BABYLON.Vector3(0,0,1));
        if (this.vecLengthSq(edgePerp) < this.epsilon) edgePerp = this.vecCross(edgeVec, new BABYLON.Vector3(0,1,0));
        edgePerp = this.vecNormalize(edgePerp);
        const refPoint = this.vecAdd(bottomPoint, this.vecScale(edgePerp, 0.1));

        for (let i = 0; i < this.vertexList.length; ++i) {
            if (i === bottomIdx || i === secondIdx) continue;
            const vol = Math.abs(this.signedVolume(bottomPoint, secondPoint, this.vertexList[i], refPoint));
            if (vol > maxVolume) { maxVolume = vol; thirdIdx = i; }
        }
        if (thirdIdx === -1) { console.warn("Jarvis March: Points are collinear"); return; }
        let thirdPoint = this.vertexList[thirdIdx];

        // orientation check using a test point
        // The normal should point OUTWARD (away from interior points)
        // If testVol > 0, the test point is in front (outside), meaning normal points inward - FLIP IT
        // If testVol < 0, the test point is behind (inside), meaning normal points outward - KEEP IT
        let testPoint = null;
        for (let i = 0; i < this.vertexList.length; ++i) {
            if (i !== bottomIdx && i !== secondIdx && i !== thirdIdx) { testPoint = this.vertexList[i]; break; }
        }
        if (testPoint) {
            const testVol = this.signedVolume(bottomPoint, secondPoint, thirdPoint, testPoint);
            if (testVol > 0) {  // FIXED: was "testVol < 0"
                // Test point is in front, so normal points inward - swap to flip it
                const tmp = secondPoint; secondPoint = thirdPoint; thirdPoint = tmp;
                const tmpIdx = secondIdx; secondIdx = thirdIdx; thirdIdx = tmpIdx;
            }
        }

        // initial face (lenient)
        let initialFaceValid = this.isValidHullFace(bottomPoint, secondPoint, thirdPoint);
        if (!initialFaceValid) {
            console.warn("Jarvis March: Initial face validation failed - continuing leniently");
            initialFaceValid = true;
        }
        if (!initialFaceValid) { console.error("Jarvis March: Cannot create initial face - aborting"); return; }

        const initialFace = new Face(this.step);
        initialFace.buildFromPoints(bottomPoint, secondPoint, thirdPoint);
        this.faces.push(initialFace);
        this.faceCreationOrder.push(0);
        processedFaces.add(this.faceKey(bottomPoint, secondPoint, thirdPoint));

        const pushEdge = (v1, v2) => {
            let found = false;
            for (const [a,b] of edgeQueue) {
                if ((this.vecEquals(a,v1) && this.vecEquals(b,v2)) || (this.vecEquals(a,v2) && this.vecEquals(b,v1))) { found = true; break; }
            }
            if (!found) edgeQueue.push([v1,v2]);
        };
        pushEdge(bottomPoint, secondPoint); pushEdge(secondPoint, thirdPoint); pushEdge(thirdPoint, bottomPoint);

        const maxIterations = this.vertexList.length * this.vertexList.length;
        let iterations = 0;
        let skippedEdges = 0;
        while (edgeQueue.length > 0 && iterations < maxIterations) {
            iterations++;
            const [e1, e2] = edgeQueue.shift();

            // find face that contains this edge (for orientation)
            let currentFace = null;
            for (const face of this.faces) {
                if (!face.points || face.points.length < 3) continue;
                const p = face.points;
                const containsEdge =
                    (this.vecEquals(p[0], e1) && this.vecEquals(p[1], e2)) ||
                    (this.vecEquals(p[1], e1) && this.vecEquals(p[2], e2)) ||
                    (this.vecEquals(p[2], e1) && this.vecEquals(p[0], e2)) ||
                    (this.vecEquals(p[0], e2) && this.vecEquals(p[1], e1)) ||
                    (this.vecEquals(p[1], e2) && this.vecEquals(p[2], e1)) ||
                    (this.vecEquals(p[2], e2) && this.vecEquals(p[0], e1));
                if (containsEdge) { currentFace = face; break; }
            }

            const edgeFaceCount = this.countFacesWithEdge(e1, e2);
            if (edgeFaceCount >= 2) {
                skippedEdges++;
                continue;
            }

            const nextIdx = this.findNextFacePoint(e1, e2, currentFace ? currentFace.normal : null);

            if (nextIdx === -1) {
                // lenient fallback: search any valid candidate
                let foundAny = false;
                for (let i = 0; i < this.vertexList.length; ++i) {
                    const p = this.vertexList[i];
                    if (this.vecEquals(p, e1) || this.vecEquals(p, e2)) continue;

                    let pointOnFaceWithEdge = false;
                    for (const face of this.faces) {
                        if (!face.points || face.points.length < 3) continue;
                        const fp1 = face.points[0], fp2 = face.points[1], fp3 = face.points[2];
                        const edgeInFace =
                            ((this.vecEquals(fp1, e1) && this.vecEquals(fp2, e2)) ||
                             (this.vecEquals(fp2, e1) && this.vecEquals(fp3, e2)) ||
                             (this.vecEquals(fp3, e1) && this.vecEquals(fp1, e2)) ||
                             (this.vecEquals(fp1, e2) && this.vecEquals(fp2, e1)) ||
                             (this.vecEquals(fp2, e2) && this.vecEquals(fp3, e1)) ||
                             (this.vecEquals(fp3, e2) && this.vecEquals(fp1, e1)));
                        if (edgeInFace && (this.vecEquals(p, fp1) || this.vecEquals(p, fp2) || this.vecEquals(p, fp3))) { pointOnFaceWithEdge = true; break; }
                    }
                    if (pointOnFaceWithEdge) continue;

                    // create face and add
                    let insideRefPoint = null;
                    if (this.faces.length > 0) {
                        let sum = new BABYLON.Vector3(0,0,0), cnt = 0;
                        for (const f of this.faces) {
                            if (f.points && f.points.length >=3) {
                                for (const pt of f.points) { sum = this.vecAdd(sum, pt); cnt++; }
                            }
                        }
                        if (cnt>0) insideRefPoint = this.vecScale(sum, 1.0/cnt);
                    }
                    if (!insideRefPoint) {
                        const edgeMid = this.vecScale(this.vecAdd(e1,e2), 0.5);
                        insideRefPoint = this.vecSubtract(edgeMid, (currentFace && currentFace.normal) ? this.vecScale(this.vecNormalize(currentFace.normal), 0.1) : new BABYLON.Vector3(0,0,0));
                    }

                    let faceP1 = e1, faceP2 = e2;
                    const testVol = this.signedVolume(faceP1, faceP2, p, insideRefPoint);
                    // If insideRefPoint is in front (testVol > 0), normal points inward - swap to fix
                    if (testVol > 0) { faceP1 = e2; faceP2 = e1; }

                    const face = new Face(this.step);
                    face.buildFromPoints(faceP1, faceP2, p);
                    this.faces.push(face);
                    this.faceCreationOrder.push(this.faces.length-1);
                    processedFaces.add(this.faceKey(faceP1, faceP2, p));

                    const addEdgeToQ = (v1, v2) => {
                        const faceCount = this.countFacesWithEdge(v1, v2);
                        if (faceCount < 2) {
                            let seenIn = false;
                            for (const [qa,qb] of edgeQueue) {
                                if ((this.vecEquals(qa, v1) && this.vecEquals(qb, v2)) || (this.vecEquals(qa, v2) && this.vecEquals(qb, v1))) { seenIn = true; break; }
                            }
                            if (!seenIn) {
                                edgeQueue.push([v1, v2]);
                            }
                        }
                    };
                    addEdgeToQ(faceP2, p); addEdgeToQ(p, faceP1);
                    foundAny = true;
                    break;
                }
                if (!foundAny) continue;
                else continue;
            }

            const nextPoint = this.vertexList[nextIdx];

            // orientation via centroid ref
            let insideRefPoint = null;
            if (this.faces.length > 0) {
                let sum = new BABYLON.Vector3(0,0,0), cnt = 0;
                for (const f of this.faces) {
                    if (f.points && f.points.length >= 3) {
                        for (const pt of f.points) { sum = this.vecAdd(sum, pt); cnt++; }
                    }
                }
                if (cnt > 0) insideRefPoint = this.vecScale(sum, 1.0/cnt);
            }
            if (!insideRefPoint) {
                const edgeMid = this.vecScale(this.vecAdd(e1,e2), 0.5);
                insideRefPoint = this.vecSubtract(edgeMid, (currentFace && currentFace.normal) ? this.vecScale(this.vecNormalize(currentFace.normal), 0.1) : new BABYLON.Vector3(0,0,0));
            }

            let faceP1 = e1, faceP2 = e2, faceP3 = nextPoint;
            const testVol = this.signedVolume(faceP1, faceP2, faceP3, insideRefPoint);
            // If insideRefPoint is in front (testVol > 0), normal points inward - swap to fix
            if (testVol > 0) { faceP1 = e2; faceP2 = e1; }

            const fKey = this.faceKey(faceP1, faceP2, faceP3);
            if (processedFaces.has(fKey)) continue;

            const face = new Face(this.step);
            face.buildFromPoints(faceP1, faceP2, faceP3);
            this.faces.push(face);
            this.faceCreationOrder.push(this.faces.length - 1);
            processedFaces.add(fKey);

            const addEdgeToQueue = (v1, v2) => {
                const faceCount = this.countFacesWithEdge(v1, v2);
                if (faceCount < 2) {
                    let already = false;
                    for (const [q1, q2] of edgeQueue) {
                        if ((this.vecEquals(q1, v1) && this.vecEquals(q2, v2)) || (this.vecEquals(q1, v2) && this.vecEquals(q2, v1))) { already = true; break; }
                    }
                    if (!already) {
                        edgeQueue.push([v1, v2]);
                    }
                }
            };
            addEdgeToQueue(faceP2, faceP3);
            addEdgeToQueue(faceP3, faceP1);
        }

        // dedupe and clean
        const uniqueFaces = [];
        const seenFaces = new Set();
        for (const face of this.faces) {
            if (!face.points || face.points.length < 3) continue;
            const fKey = this.faceKey(face.points[0], face.points[1], face.points[2]);
            if (!seenFaces.has(fKey)) { seenFaces.add(fKey); uniqueFaces.push(face); }
        }
        this.faces = uniqueFaces;

        console.log(`Jarvis March: Found ${this.faces.length} faces after ${iterations} iterations (${skippedEdges} edges skipped, ${edgeQueue.length} edges remaining)`);

        // run cleaning and validation passes
        this.cleanFaces();                 // remove degenerate/duplicate triangles
        this.removeCoplanarFaces();        // remove nearly coplanar faces (z-fighting prevention)
        this.validateAndOrientFaces();     // orient flips or remove internal faces

        if (this.faces.length === 0) {
            console.error("Jarvis March: WARNING - No faces computed! Check input geometry.");
            console.error(`  Input points: ${this.vertexList.length}`);
        }
        // end compute
    }

    // ---------- Rendering ----------
    buildRenderableMesh(scene, singleColor = null) {
        console.log(`Jarvis March: Building renderable mesh with ${this.faces.length} faces`);

        if (this.renderableMesh) {
            try { this.renderableMesh.dispose(); } catch (e) { /* ignore */ }
            this.renderableMesh = null;
        }
        this.renderableMesh = new BABYLON.Mesh("jarvis-march-hull", scene);

        if (!this.faces || this.faces.length === 0) {
            console.warn("Jarvis March: No faces to render - creating empty mesh");
            this.renderableMesh.setVerticesData(BABYLON.VertexBuffer.PositionKind, [], false);
            this.renderableMesh.setIndices([]);
            this.constructionAnimation = new BABYLON.AnimationGroup("jarvis-march-anim", scene);
            return;
        }

        const vertexData = new BABYLON.VertexData();

        // Compute interior reference point for consistent normal orientation
        // Use average of input points (which should be inside the hull)
        let interiorPoint = new BABYLON.Vector3(0, 0, 0);
        if (this.vertexList && this.vertexList.length > 0) {
            for (let v of this.vertexList) {
                interiorPoint = this.vecAdd(interiorPoint, v);
            }
            interiorPoint = this.vecScale(interiorPoint, 1.0 / this.vertexList.length);
        } else {
            // Fallback: use centroid of face vertices
            let totalVerts = 0;
            for (let f of this.faces) {
                for (let v of f.points) {
                    interiorPoint = this.vecAdd(interiorPoint, v);
                    totalVerts++;
                }
            }
            if (totalVerts > 0) {
                interiorPoint = this.vecScale(interiorPoint, 1.0 / totalVerts);
            }
        }

        // EXACTLY match QuickHull's approach: duplicate vertices, no sharing
        const vertices = []; 
        const faces = [];
        const normals = [];
        const lifetime = [];
        
        let faceIndex = 0;
        for (let f of this.faces) {
            const v1 = f.points[0], v2 = f.points[1], v3 = f.points[2];
            
            // Compute normal from original face points
            const edge1 = this.vecSubtract(v2, v1);
            const edge2 = this.vecSubtract(v3, v1);
            let normal = this.vecCross(edge1, edge2);
            normal = this.vecNormalize(normal);
            
            // Ensure normal points outward (away from interior point)
            const faceCenter = this.vecScale(this.vecAdd(this.vecAdd(v1, v2), v3), 1.0/3);
            const toInterior = this.vecSubtract(interiorPoint, faceCenter);
            const toInteriorLen = Math.sqrt(this.vecLengthSq(toInterior));
            
            // Use a larger epsilon for orientation check (relative to face size)
            const orientationEpsilon = Math.max(this.epsilon * 100, toInteriorLen * 1e-6);
            const dot = this.vecDot(normal, toInterior);
            
            // Determine vertex order based on normal orientation
            let pointsToUse = f.points;
            if (dot > orientationEpsilon) {
                // Normal points inward, reverse vertex order
                pointsToUse = [f.points[0], f.points[2], f.points[1]]; // Reverse winding
                // Recompute normal from flipped vertices to ensure consistency
                const flippedEdge1 = this.vecSubtract(pointsToUse[1], pointsToUse[0]);
                const flippedEdge2 = this.vecSubtract(pointsToUse[2], pointsToUse[0]);
                normal = this.vecCross(flippedEdge1, flippedEdge2);
                normal = this.vecNormalize(normal);
            }
            
            // Push all 3 vertices for this face first
            const startIdx = vertices.length / 3;
            for (let v of pointsToUse) {
                vertices.push(v.x, v.y, v.z);
                // Use the same corrected normal for all 3 vertices of the face
                normals.push(normal.x, normal.y, normal.z);
            }
            // Then push indices once per face (counter-clockwise winding)
            faces.push(startIdx, startIdx + 1, startIdx + 2);
            
            // Use face index as creation time for sequential animation
            lifetime.push({createdAt: faceIndex, deletedAt: null});
            faceIndex++;
        }

        const materials = [];
        const multimaterial = new BABYLON.MultiMaterial("jarvis-hull", scene);
        
        const animations = new BABYLON.AnimationGroup("jarvis-march-anim", scene);

        const START_OPACITY = 0.0;
        const END_OPACITY = CONSTS.HULL_OPACITY;

        const colArr = [];

        // Create materials and animations for each face
        for (let i = 0; i < lifetime.length; i++) {
            const {createdAt, deletedAt} = lifetime[i];

            const nm = new BABYLON.StandardMaterial("jarvis-face" + i);
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

            const animation = new BABYLON.Animation("jarvis-face" + i, "alpha", CONSTS.FPS, BABYLON.Animation.ANIMATIONTYPE_FLOAT);
            const keys = [];
            keys.push(
                {frame: 0, value: 0},
                {frame: createdAt*CONSTS.FPS, value: START_OPACITY},
                {frame: (createdAt+1)*CONSTS.FPS, value: END_OPACITY}
            );

            if (deletedAt) {
                keys.push(
                    {frame: deletedAt*CONSTS.FPS, value: END_OPACITY},
                    {frame: (deletedAt+1)*CONSTS.FPS, value: START_OPACITY}
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
            new BABYLON.SubMesh(i, 0, vertices.length, i*3, 3, this.renderableMesh);
        }

        this.renderableMesh.material = multimaterial;

        this.constructionAnimation = animations;
        console.log(`Jarvis March: Built mesh with ${vertices.length/3} vertices and ${faces.length/3} triangles (${lifetime.length} faces)`);
    }
}
