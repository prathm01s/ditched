#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// Graham Scan Algorithm for Computing Convex Hull
// Time Complexity: O(n log n) due to sorting
// Space Complexity: O(n) for storing hull points

// --- DATA STRUCTURES ---

// Point structure representing a 2D point
struct pt {
    double x, y;
    bool operator == (pt const& t) const { return x == t.x && y == t.y; }
};

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

// Check if points are collinear
bool collinear(pt a, pt b, pt c) { return orientation(a, b, c) == 0; }

// --- GRAHAM SCAN CONVEX HULL ALGORITHM ---

// Computes the convex hull using Graham's scan algorithm
// Input: Vector of points (modified in-place)
// Output: Points vector is replaced with hull points in counter-clockwise order
// Algorithm Steps:
//   1. Find lowest point (min y, then min x) as pivot p0
//   2. Sort all points by polar angle with respect to p0
//   3. Use a stack to build hull by maintaining counter-clockwise turns
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.empty()) return; // Safety check
    
    // Step 1: Find pivot point p0 (lowest y-coordinate, leftmost if tie)
    pt p0 = *min_element(a.begin(), a.end(), [](pt a, pt b) {
        return make_pair(a.y, a.x) < make_pair(b.y, b.x);
    });
    
    // Step 2: Sort points by polar angle with respect to p0
    // If angles are equal, sort by distance from p0
    sort(a.begin(), a.end(), [&p0](const pt& a, const pt& b) {
        int o = orientation(p0, a, b);
        if (o == 0)
            return (p0.x-a.x)*(p0.x-a.x) + (p0.y-a.y)*(p0.y-a.y)
                < (p0.x-b.x)*(p0.x-b.x) + (p0.y-b.y)*(p0.y-b.y);
        return o < 0;
    });
    
    // Handle collinear points on the last segment if requested
    if (include_collinear) {
        int i = (int)a.size()-1;
        while (i >= 0 && collinear(p0, a[i], a.back())) i--;
        reverse(a.begin()+i+1, a.end());
    }
    
    // Step 3: Build hull using a stack
    vector<pt> st;
    for (int i = 0; i < (int)a.size(); i++) {
        // Remove points that would create a clockwise turn
        while (st.size() > 1 && !cw(st[st.size()-2], st.back(), a[i], include_collinear))
            st.pop_back();
        st.push_back(a[i]);
    }
    
    // Edge case: if all points are collinear and form a line segment
    if (include_collinear == false && st.size() == 2 && st[0] == st[1])
        st.pop_back();
    
    a = st;
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
        getline(file, line); // Consume header line
    }

    // 2. READ DATA
    // Parse comma-separated x,y coordinate pairs
    // Handles formats: "123.45,678.90" or "123.45 678.90"
    while (file >> x) {
        char c;
        // Eat potential comma separator
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) {
            points.push_back({x, y});
        }
    }
    file.close();

    // Edge case: Need at least 3 points for a convex hull
    if (points.size() < 3) {
        // Return 0 time if not enough points to hull
        cerr << "BENCHMARK_TIME_SEC=0.0" << endl;
        cerr << "BENCHMARK_MEM_KB=0" << endl;
        return 0;
    }

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
    // BENCHMARK_TIME_SEC: Algorithm execution time in seconds
    // BENCHMARK_MEM_KB: Peak memory usage in kilobytes
    cerr << "BENCHMARK_TIME_SEC=" << fixed << setprecision(9) << elapsed.count() << endl;
    cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << endl;
    
    return 0;
}