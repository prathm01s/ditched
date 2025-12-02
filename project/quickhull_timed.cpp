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

#ifdef _OPENMP
    #include <omp.h>
#else
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

namespace QuickHull {
    // [KEEP ALL YOUR QUICKHULL NAMESPACE CODE EXACTLY AS IS]
    // [INSERT QUICKHULL ALGORITHMS HERE]
    struct Point {
        double x, y;
        Point() : x(0), y(0) {}
        Point(double x_, double y_) : x(x_), y(y_) {}
        bool operator==(const Point& other) const {
            constexpr double EPS = 1e-12;
            return std::abs(x - other.x) < EPS && std::abs(y - other.y) < EPS;
        }
        Point operator-(const Point& other) const {
            return Point(x - other.x, y - other.y);
        }
    };
    inline double crossProduct(const Point& p1, const Point& p2, const Point& p3) {
        return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    }
    inline double signedDistanceSq(const Point& p, const Point& lineStart, const Point& lineEnd) {
        double dx = lineEnd.x - lineStart.x;
        double dy = lineEnd.y - lineStart.y;
        return (p.x - lineStart.x) * dy - (p.y - lineStart.y) * dx;
    }
    inline int findSide(const Point& p1, const Point& p2, const Point& p) {
        double val = (p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x);
        constexpr double EPS = 1e-12;
        if (val > EPS) return 1;
        if (val < -EPS) return -1;
        return 0;
    }
    inline double distanceFromLine(const Point& p1, const Point& p2, const Point& p) {
        return std::abs((p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x));
    }
    void quickHullRecursive(const std::vector<Point>& points,
                           const Point& P, const Point& Q,
                           int side,
                           std::vector<Point>& hull) {
        constexpr double EPS = 1e-12;
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
        if (farthestIdx < 0 || maxDist <= EPS) {
            hull.push_back(Q);
            return;
        }
        const Point& C = points[farthestIdx];
        quickHullRecursive(points, P, C, -findSide(P, C, Q), hull);
        quickHullRecursive(points, C, Q, -findSide(C, Q, P), hull);
    }
    std::vector<Point> computeConvexHull(const std::vector<Point>& inputPoints) {
        if (inputPoints.size() < 3) return inputPoints;
        std::vector<Point> points = inputPoints;
        std::sort(points.begin(), points.end(), 
                  [](const Point& a, const Point& b) {
                      return (a.x < b.x) || (a.x == b.x && a.y < b.y);
                  });
        points.erase(std::unique(points.begin(), points.end()), points.end());
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
        std::vector<Point> hull;
        hull.reserve(points.size());
        hull.push_back(leftmost);
        quickHullRecursive(points, leftmost, rightmost, 1, hull);
        quickHullRecursive(points, rightmost, leftmost, 1, hull);
        return hull;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    std::ifstream file(argv[1]);
    if (!file.is_open()) return 1;
    
    std::vector<QuickHull::Point> all_points;
    double x, y;
    std::string line;
    // Skip Header
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }
    // Read Data
    while (file >> x) {
        if (file.peek() == ',') file.ignore();
        if (file >> y) all_points.push_back({x, y});
    }
    file.close();

    struct rusage usage_end;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto hull = QuickHull::computeConvexHull(all_points);
    auto end_time = std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_end);

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cerr << "BENCHMARK_TIME_SEC=" << std::fixed << std::setprecision(9) << elapsed.count() << std::endl;
    std::cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << std::endl;
    return 0;
}