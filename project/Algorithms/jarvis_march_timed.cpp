#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// --- DATA STRUCTURES ---
struct pt {
    double x, y;
    // Overload == for easy comparison
    bool operator==(const pt& other) const {
        return x == other.x && y == other.y;
    }
};

// --- GEOMETRIC HELPERS ---

// returns squared Euclidean distance to avoid sqrt overhead
double distSq(pt p1, pt p2) {
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

// 0 -> collinear, 1 -> clockwise, 2 -> counter-clockwise
// Uses the same cross-product logic as your Monotone Chain implementation
int orientation(pt a, pt b, pt c) {
    double val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);
    if (abs(val) < 1e-9) return 0;  // Collinear
    return (val > 0) ? 1 : 2;       // Clockwise or Counter-Clockwise
}

// --- JARVIS MARCH ALGORITHM ---
void convex_hull(vector<pt>& points) {
    int n = points.size();
    if (n < 3) return; // Hull is just the points themselves if n < 3

    vector<pt> hull;

    // Step 1: Find the leftmost point (min x), break ties with min y
    int l = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].x < points[l].x || (points[i].x == points[l].x && points[i].y < points[l].y))
            l = i;
    }

    // Step 2: Wrap the points
    int p = l;
    int q;
    do {
        hull.push_back(points[p]);

        // Search for a point 'q' such that orientation(p, q, x) is counter-clockwise 
        // for all other points 'x'.
        q = (p + 1) % n;
        
        for (int i = 0; i < n; i++) {
            // If i is more counter-clockwise than current q, then update q
            int o = orientation(points[p], points[i], points[q]);

            if (o == 2) { 
                q = i;
            }
            // Robustness: Handle collinear points. 
            // If points[i] is collinear with line pq, usually we want the farthest one 
            // to minimize edge count and ensure the hull encloses all points.
            else if (o == 0) {
                if (distSq(points[p], points[i]) > distSq(points[p], points[q])) {
                    q = i;
                }
            }
        }

        // p becomes the point we just found
        p = q;

    } while (p != l); // Stop when we return to the start point

    points = hull;
}

// --- BENCHMARKING MAIN FUNCTION ---
int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    ifstream file(argv[1]);
    if (!file.is_open()) return 1;

    vector<pt> points;
    double x, y;
    string line;
    
    // 1. SKIP HEADER (Robust skip for non-numeric lines)
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }

    // 2. READ DATA (Handles comma separation if present)
    while (file >> x) {
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) {
            points.push_back({x, y});
        }
    }
    file.close();

    // 3. HANDLE DEGENERATE CASES
    if (points.size() < 3) {
        // Just return 0 stats for degenerate cases
        cerr << "BENCHMARK_TIME_SEC=0.0" << endl;
        cerr << "BENCHMARK_MEM_KB=0" << endl;
        return 0;
    }

    // 4. EXECUTE AND TIME ALGORITHM
    struct rusage usage_end;
    auto start_time = chrono::high_resolution_clock::now();

    convex_hull(points);

    auto end_time = chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_end);

    chrono::duration<double> elapsed = end_time - start_time;
    
    // 5. OUTPUT METRICS TO STDERR
    cerr << "BENCHMARK_TIME_SEC=" << fixed << setprecision(9) << elapsed.count() << endl;
    cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << endl;
    
    return 0;
}