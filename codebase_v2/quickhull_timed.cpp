#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <fstream>   
#include <iostream>  
#include <string>    
#include <iomanip>   
#include <stdexcept> 
#include <sys/stat.h> 
#include <chrono>
#include <sys/resource.h>

// OpenMP support (optional - for parallel processing)
#ifdef _OPENMP
    #include <omp.h>
#else
    // Provide dummy definitions if OpenMP is not available
    inline int omp_get_num_threads() { return 1; }
    inline int omp_get_thread_num() { return 0; }
    #define omp_get_num_threads() 1
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
    #define omp_set_num_threads(x)
    #define omp_parallel
    #define omp_for for
    #define omp_critical
    #define omp_parallel_for for
    #define omp_reduction(op, var)
#endif

// QuickHull Algorithm for Computing Convex Hull
// Time Complexity: O(n log n) average case, O(n²) worst case
// Space Complexity: O(n) for recursion and hull storage
// Divide-and-conquer approach - works well when hull has few points

namespace QuickHull {
    
    // --- DATA STRUCTURES ---
    
    // Point structure representing a 2D point
    struct Point {
        double x, y;
        Point() : x(0), y(0) {}
        Point(double x_, double y_) : x(x_), y(y_) {}
        
        // Equality comparison with epsilon tolerance
        bool operator==(const Point& other) const {
            constexpr double EPS = 1e-12;
            return std::abs(x - other.x) < EPS && std::abs(y - other.y) < EPS;
        }
        
        // Vector subtraction
        Point operator-(const Point& other) const {
            return Point(x - other.x, y - other.y);
        }
    };
    
    // --- GEOMETRIC HELPER FUNCTIONS ---
    
    // Compute cross product of vectors (p2-p1) and (p3-p1)
    // Returns positive value if p3 is to the left of line p1->p2
    inline double crossProduct(const Point& p1, const Point& p2, const Point& p3) {
        return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    }
    
    // Compute signed distance squared from point p to line (lineStart -> lineEnd)
    // Sign indicates which side of the line the point is on
    inline double signedDistanceSq(const Point& p, const Point& lineStart, const Point& lineEnd) {
        double dx = lineEnd.x - lineStart.x;
        double dy = lineEnd.y - lineStart.y;
        return (p.x - lineStart.x) * dy - (p.y - lineStart.y) * dx;
    }
    
    // Determine which side of a line the point p is on
    // Returns: +1 (left), -1 (right), 0 (on line)
    inline int findSide(const Point& p1, const Point& p2, const Point& p) {
        double val = (p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x);
        constexpr double EPS = 1e-12;
        if (val > EPS) return 1;
        if (val < -EPS) return -1;
        return 0;
    }
    
    // Calculate perpendicular distance from point p to line (p1 -> p2)
    inline double distanceFromLine(const Point& p1, const Point& p2, const Point& p) {
        return std::abs((p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x));
    }
    
    // --- QUICKHULL RECURSIVE ALGORITHM ---
    
    // Recursive function to find hull points on one side of a line segment
    // Params:
    //   points: All input points
    //   P, Q: Endpoints of current line segment
    //   side: Which side of the line to process (+1 or -1)
    //   hull: Output vector to accumulate hull points
    void quickHullRecursive(const std::vector<Point>& points,
                           const Point& P, const Point& Q,
                           int side,
                           std::vector<Point>& hull) {
        constexpr double EPS = 1e-12;
        
        // Find the farthest point from line PQ on the given side
        int farthestIdx = -1;
        double maxDist = 0.0;
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (findSide(P, Q, points[i]) == side) {
                double dist = distanceFromLine(P, Q, points[i]);
                if (dist > maxDist) {
                    maxDist = dist;
                    farthestIdx = static_cast<int>(i);
                }
            }
        }
        
        // Base case: No points on this side or all points are on the line
        if (farthestIdx < 0 || maxDist <= EPS) {
            hull.push_back(Q);
            return;
        }
        
        // Recursive case: Found farthest point C
        // Divide into two sub-problems: PC and CQ
        const Point& C = points[farthestIdx];
        
        // Recursively process triangle PCC
        quickHullRecursive(points, P, C, -findSide(P, C, Q), hull);
        // Recursively process triangle CQ
        quickHullRecursive(points, C, Q, -findSide(C, Q, P), hull);
    }
    
    // --- MAIN QUICKHULL ALGORITHM ---
    
    // Compute convex hull using QuickHull algorithm
    // Returns vector of hull points in counter-clockwise order
    std::vector<Point> computeConvexHull(const std::vector<Point>& inputPoints) {
        // Edge case: fewer than 3 points
        if (inputPoints.size() < 3) return inputPoints;
        
        // Make a copy and sort to remove duplicates
        std::vector<Point> points = inputPoints;
        std::sort(points.begin(), points.end(), 
                  [](const Point& a, const Point& b) {
                      return (a.x < b.x) || (a.x == b.x && a.y < b.y);
                  });
        points.erase(std::unique(points.begin(), points.end()), points.end());
        
        // Find leftmost and rightmost points (guaranteed to be on hull)
        auto leftmostIt = std::min_element(points.begin(), points.end(),
                                           [](const Point& a, const Point& b) {
                                               return (a.x < b.x) || (a.x == b.x && a.y < b.y);
                                           });
        auto rightmostIt = std::max_element(points.begin(), points.end(),
                                            [](const Point& a, const Point& b) {
                                                return (a.x < b.x) || (a.x == b.x && a.y < b.y);
                                            });
        Point leftmost = *leftmostIt;
        Point rightmost = *rightmostIt;
        
        // Initialize hull with leftmost point
        std::vector<Point> hull;
        hull.reserve(points.size());
        hull.push_back(leftmost);
        
        // Recursively find hull points above and below the line leftmost->rightmost
        // Upper hull (side = 1): points above the line
        quickHullRecursive(points, leftmost, rightmost, 1, hull);
        // Lower hull (side = 1 from rightmost to leftmost): points below the line
        quickHullRecursive(points, rightmost, leftmost, 1, hull);
        
        return hull;
    }
}
                                            [](const Point& a, const Point& b) {
                                                return (a.x < b.x) || (a.x == b.x && a.y < b.y);
                                            });
        Point leftmost = *leftmostIt;
        Point rightmost = *rightmostIt;
        std::vector<Point> hull;
        hull.reserve(points.size());
        hull.push_back(leftmost);
        quickHullRecursive(points, leftmost, rightmost, 1, hull);
        quickHullRecursive(points, rightmost, leftmost, 1, hull);
        return hull;
    }
}

// --- BENCHMARKING MAIN FUNCTION ---

int main(int argc, char* argv[]) {
    // Validate command-line arguments
    if (argc != 2) return 1;
    
    // Open input file
    std::ifstream file(argv[1]);
    if (!file.is_open()) return 1;
    
    std::vector<QuickHull::Point> all_points;
    double x, y;
    std::string line;
    
    // Skip Header - Skip any non-numeric header lines
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }
    
    // Read Data - Parse comma-separated x,y coordinate pairs
    while (file >> x) {
        if (file.peek() == ',') file.ignore();
        if (file >> y) all_points.push_back({x, y});
    }
    file.close();

    // Memory usage tracking structure
    struct rusage usage_end;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // *** CORE ALGORITHM EXECUTION ***
    auto hull = QuickHull::computeConvexHull(all_points);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Get memory usage statistics
    getrusage(RUSAGE_SELF, &usage_end);

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Output results to stderr for benchmark script parsing
    std::cerr << "BENCHMARK_TIME_SEC=" << std::fixed << std::setprecision(9) << elapsed.count() << std::endl;
    std::cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << std::endl;
    
    return 0;
}