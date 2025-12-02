#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// Optimized Monotone Chain Algorithm for Computing Convex Hull
// Time Complexity: O(n log n) worst-case, O(n) best-case for pre-sorted data
// Space Complexity: O(n) for storing hull points
// Optimization: Checks if data is already sorted before sorting

// --- DATA STRUCTURES ---

// Point structure representing a 2D point
struct pt { double x, y; };

// --- GEOMETRIC HELPER FUNCTIONS ---

// Compute the orientation of ordered triplet (a, b, c)
// Returns: -1 (clockwise), +1 (counter-clockwise), 0 (collinear)
int orientation(pt a, pt b, pt c) {
    double v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y);
    if (v < 0) return -1; 
    if (v > 0) return +1; 
    return 0; 
}

// Check if points make a clockwise turn
bool cw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}

// Check if points make a counter-clockwise turn
bool ccw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o > 0 || (include_collinear && o == 0);
}

// --- OPTIMIZED MONOTONE CHAIN ALGORITHM ---

// Optimized version that avoids sorting if data is already sorted
// This can improve performance for datasets that are naturally ordered
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.size() <= 2) return;
    
    // Define lexicographical comparison function
    auto lexicographical_compare = [](pt a, pt b) {
        return make_pair(a.x, a.y) < make_pair(b.y, b.y);
    };
    
    // OPTIMIZATION: Check if already sorted before sorting
    // For pre-sorted data, this saves O(n log n) time
    if (!is_sorted(a.begin(), a.end(), lexicographical_compare)) {
        sort(a.begin(), a.end(), lexicographical_compare);
    }
    
    // Get leftmost and rightmost points
    pt p1 = a[0], p2 = a.back();
    
    // Build upper and lower hulls
    vector<pt> up, down;
    up.push_back(p1);
    down.push_back(p1);
    
    for (int i = 1; i < (int)a.size(); i++) {
        // Upper hull construction
        if (i == a.size() - 1 || cw(p1, a[i], p2, include_collinear)) {
            while (up.size() >= 2 && !cw(up[up.size()-2], up[up.size()-1], a[i], include_collinear))
                up.pop_back();
            up.push_back(a[i]);
        }
        // Lower hull construction
        if (i == a.size() - 1 || ccw(p1, a[i], p2, include_collinear)) {
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
    
    // Concatenate hulls
    a.clear();
    for (int i = 0; i < (int)up.size(); i++) a.push_back(up[i]);
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
    // The optimized version checks if sorted before sorting
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