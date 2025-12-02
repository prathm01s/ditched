#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// Monotone Chain Algorithm (Andrew's Algorithm) for Computing Convex Hull
// Time Complexity: O(n log n) due to sorting
// Space Complexity: O(n) for storing hull points
// More numerically stable than Graham Scan

// --- DATA STRUCTURES ---

// Point structure representing a 2D point
struct pt { double x, y; };

// --- GEOMETRIC HELPER FUNCTIONS ---

// Compute the orientation of ordered triplet (a, b, c)
// Returns:
//   -1: Clockwise orientation
//   +1: Counter-clockwise orientation
//    0: Collinear points
// Uses the cross product: (b-a) × (c-a)
int orientation(pt a, pt b, pt c) {
    double v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y);
    if (v < 0) return -1; 
    if (v > 0) return +1; 
    return 0; 
}

// Check if points a, b, c make a clockwise turn (or are collinear if include_collinear=true)
bool cw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}

// Check if points a, b, c make a counter-clockwise turn (or are collinear if include_collinear=true)
bool ccw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o > 0 || (include_collinear && o == 0);
}

// --- MONOTONE CHAIN CONVEX HULL ALGORITHM ---

// Computes the convex hull using Andrew's Monotone Chain algorithm
// Input: Vector of points (modified in-place)
// Output: Points vector is replaced with hull points in counter-clockwise order
// Algorithm Steps:
//   1. Sort points lexicographically (first by x, then by y)
//   2. Build lower hull (left to right)
//   3. Build upper hull (right to left)
//   4. Concatenate hulls
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.size() <= 2) return;
    
    // Step 1: Sort points lexicographically by (x, y)
    sort(a.begin(), a.end(), [](pt a, pt b) {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    });
    
    // Get leftmost and rightmost points
    pt p1 = a[0], p2 = a.back();
    
    // Step 2 & 3: Build upper and lower hulls simultaneously
    vector<pt> up, down;
    up.push_back(p1);
    down.push_back(p1);
    
    for (int i = 1; i < (int)a.size(); i++) {
        // Build upper hull (points above or on the line p1->p2)
        if (i == a.size() - 1 || cw(p1, a[i], p2, include_collinear)) {
            // Remove points that create a counter-clockwise turn
            while (up.size() >= 2 && !cw(up[up.size()-2], up[up.size()-1], a[i], include_collinear))
                up.pop_back();
            up.push_back(a[i]);
        }
        
        // Build lower hull (points below or on the line p1->p2)
        if (i == a.size() - 1 || ccw(p1, a[i], p2, include_collinear)) {
            // Remove points that create a clockwise turn
            while (down.size() >= 2 && !ccw(down[down.size()-2], down[down.size()-1], a[i], include_collinear))
                down.pop_back();
            down.push_back(a[i]);
        }
    }
    
    // Special case: all points are collinear
    if (include_collinear && up.size() == a.size()) {
        reverse(a.begin(), a.end());
        return;
    }
    
    // Step 4: Concatenate upper and lower hulls
    a.clear();
    // Add all points from upper hull
    for (int i = 0; i < (int)up.size(); i++) a.push_back(up[i]);
    // Add points from lower hull (excluding first and last to avoid duplicates)
    for (int i = down.size() - 2; i > 0; i--) a.push_back(down[i]);
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
    
    // Skip Header - Skip any non-numeric header lines
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }
    
    // Read Data - Parse comma-separated x,y coordinate pairs
    while (file >> x) {
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) points.push_back({x, y});
    }
    file.close();

    // Memory usage tracking structure
    struct rusage usage_end;
    
    // Start timing
    auto start_time = chrono::high_resolution_clock::now();
    
    // *** CORE ALGORITHM EXECUTION ***
    convex_hull(points, false);
    
    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    
    // Get memory usage statistics
    getrusage(RUSAGE_SELF, &usage_end);

    // Calculate elapsed time
    chrono::duration<double> elapsed = end_time - start_time;
    
    // Output results to stderr for benchmark script parsing
    cerr << "BENCHMARK_TIME_SEC=" << fixed << setprecision(9) << elapsed.count() << endl;
    cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << endl;
    
    return 0;
}