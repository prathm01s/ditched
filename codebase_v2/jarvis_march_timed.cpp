#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// Jarvis March (Gift Wrapping) Algorithm for Computing Convex Hull
// Time Complexity: O(nh) where n is number of points and h is hull size
// Space Complexity: O(h) for storing hull points
// Best Case: O(n) when h is small (few hull points)
// Worst Case: O(n²) when h = n (all points on hull)

// --- DATA STRUCTURES ---

// Point structure representing a 2D point
struct pt {
    double x, y;
    // Overload == for easy comparison
    bool operator==(const pt& other) const {
        return x == other.x && y == other.y;
    }
};

// --- GEOMETRIC HELPER FUNCTIONS ---

// Calculate squared Euclidean distance between two points
// Returns squared distance to avoid sqrt overhead
// Used for tie-breaking when points are collinear
double distSq(pt p1, pt p2) {
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

// Determine orientation of ordered triplet (a, b, c)
// Returns:
//   0: Collinear points (within tolerance)
//   1: Clockwise orientation  
//   2: Counter-clockwise orientation
// Uses cross product: (b-a) × (c-b)
int orientation(pt a, pt b, pt c) {
    double val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (abs(val) < 1e-9) return 0;  // Collinear
    return (val > 0) ? 1 : 2;       // Clockwise or Counter-Clockwise
}

// --- JARVIS MARCH (GIFT WRAPPING) ALGORITHM ---

// Computes the convex hull using Jarvis March (Gift Wrapping) algorithm
// Input: Vector of points (modified in-place)
// Output: Points vector is replaced with hull points in counter-clockwise order
// Algorithm Steps:
//   1. Start from the leftmost point
//   2. Find the next point by checking all points for the most counter-clockwise turn
//   3. Repeat until we wrap back to the starting point
void convex_hull(vector<pt>& points) {
    int n = points.size();
    if (n < 3) return; // Hull is just the points themselves if n < 3

    vector<pt> hull;

    // Step 1: Find the leftmost point (min x, break ties with min y)
    // This point is guaranteed to be on the hull
    int l = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].x < points[l].x || (points[i].x == points[l].x && points[i].y < points[l].y))
            l = i;
    }

    // Step 2: "Wrap" around the point set to find hull vertices
    // Start from leftmost point and keep moving counter-clockwise
    int p = l;
    int q;
    do {
        // Add current point to hull
        hull.push_back(points[p]);

        // Find the most counter-clockwise point from points[p]
        // Initialize q as the next point in circular order
        q = (p + 1) % n;
        
        // Check all other points to find the most counter-clockwise one
        for (int i = 0; i < n; i++) {
            // If i is more counter-clockwise than current q, then update q
            int o = orientation(points[p], points[i], points[q]);

            if (o == 2) { 
                // Point i is more counter-clockwise
                q = i;
            }
            // Robustness: Handle collinear points
            // If points[i] is collinear with line pq, choose the farthest one
            // This minimizes edge count and ensures the hull encloses all points
            else if (o == 0) {
                if (distSq(points[p], points[i]) > distSq(points[p], points[q])) {
                    q = i;
                }
            }
        }

        // Move to the found point for the next iteration
        p = q;

    } while (p != l); // Stop when we return to the start point

    // Replace input with hull points
    points = hull;
}

// --- BENCHMARKING MAIN FUNCTION ---

int main(int argc, char* argv[]) {
    // Validate command-line arguments
    if (argc != 2) return 1;
    
    // Open input file
    ifstream file(argv[1]);
    if (!file.is_open()) return 1;

    vector<pt> points;
    double x, y;
    string line;
    
    // 1. SKIP HEADER
    // Skip any non-numeric header lines (e.g., "x,y")
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }

    // 2. READ DATA
    // Parse comma-separated or space-separated x,y coordinate pairs
    while (file >> x) {
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) {
            points.push_back({x, y});
        }
    }
    file.close();

    // 3. HANDLE DEGENERATE CASES
    // Edge case: Need at least 3 points for a convex hull
    if (points.size() < 3) {
        // Just return 0 stats for degenerate cases
        cerr << "BENCHMARK_TIME_SEC=0.0" << endl;
        cerr << "BENCHMARK_MEM_KB=0" << endl;
        return 0;
    }

    // 4. EXECUTE AND TIME ALGORITHM
    // Memory usage tracking structure
    struct rusage usage_end;
    
    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    // *** CORE ALGORITHM EXECUTION ***
    convex_hull(points);

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    
    // Get memory usage statistics
    getrusage(RUSAGE_SELF, &usage_end);

    // Calculate elapsed time
    chrono::duration<double> elapsed = end_time - start_time;
    
    // 5. OUTPUT METRICS TO STDERR
    // Output results for benchmark script parsing
    // BENCHMARK_TIME_SEC: Algorithm execution time in seconds
    // BENCHMARK_MEM_KB: Peak memory usage in kilobytes
    cerr << "BENCHMARK_TIME_SEC=" << fixed << setprecision(9) << elapsed.count() << endl;
    cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << endl;
    
    return 0;
}