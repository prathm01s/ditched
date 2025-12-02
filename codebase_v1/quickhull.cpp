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
#include <sys/stat.h> // For mkdir

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

// --- YOUR QUICKHULL NAMESPACE (Complete) ---
namespace QuickHull {

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

int findFarthestPointSIMD(const std::vector<Point>& points, 
                         const Point& start, const Point& end) {
    if (points.empty()) return -1;
    
    constexpr double EPS = 1e-12;
    double maxDist = -1.0;
    int farthestIdx = -1;
    
    for (size_t i = 0; i < points.size(); ++i) {
        double cross = signedDistanceSq(points[i], start, end);
        double dist = std::abs(cross);
        
        if (dist > maxDist) {
            maxDist = dist;
            farthestIdx = static_cast<int>(i);
        }
    }
    
    return (maxDist > EPS) ? farthestIdx : -1;
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

std::vector<Point> prefilterPoints(const std::vector<Point>& points) {
    if (points.size() < 100) {
        return points;
    }
    
    double minX = points[0].x, maxX = points[0].x;
    double minY = points[0].y, maxY = points[0].y;
    
    #ifdef _OPENMP
    #pragma omp parallel
    {
        double localMinX = minX, localMaxX = maxX;
        double localMinY = minY, localMaxY = maxY;
        #pragma omp for nowait
        for (int i = 1; i < static_cast<int>(points.size()); ++i) {
            localMinX = std::min(localMinX, points[i].x);
            localMaxX = std::max(localMaxX, points[i].x);
            localMinY = std::min(localMinY, points[i].y);
            localMaxY = std::max(localMaxY, points[i].y);
        }
        #pragma omp critical
        {
            minX = std::min(minX, localMinX);
            maxX = std::max(maxX, localMaxX);
            minY = std::min(minY, localMinY);
            maxY = std::max(maxY, localMaxY);
        }
    }
    #else
    for (int i = 1; i < static_cast<int>(points.size()); ++i) {
        minX = std::min(minX, points[i].x);
        maxX = std::max(maxX, points[i].x);
        minY = std::min(minY, points[i].y);
        maxY = std::max(maxY, points[i].y);
    }
    #endif
    
    double xMargin = (maxX - minX) * 0.05;
    double yMargin = (maxY - minY) * 0.05;
    
    std::vector<Point> filtered;
    filtered.reserve(points.size());
    
    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<Point> localFiltered;
        localFiltered.reserve(points.size() / omp_get_num_threads());
        
        #pragma omp for nowait
    #else
    {
        std::vector<Point> localFiltered;
        localFiltered.reserve(points.size());
    #endif
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const Point& p = points[i];
            if (p.x <= minX + xMargin || p.x >= maxX - xMargin ||
                p.y <= minY + yMargin || p.y >= maxY - yMargin) {
                localFiltered.push_back(p);
            }
        }
        
        #ifdef _OPENMP
        #pragma omp critical
        #endif
        {
            filtered.insert(filtered.end(), localFiltered.begin(), localFiltered.end());
        }
    }
    
    return (filtered.size() >= points.size() * 0.2) ? filtered : points;
}

std::vector<Point> computeConvexHull(const std::vector<Point>& inputPoints) {
    if (inputPoints.size() < 3) {
        return inputPoints;
    }
    
    std::vector<Point> points = inputPoints;
    std::sort(points.begin(), points.end(), 
              [](const Point& a, const Point& b) {
                  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
              });
    points.erase(std::unique(points.begin(), points.end()), points.end());
    
    if (points.size() < 3) {
        return points;
    }
    
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
    
    constexpr double EPS = 1e-12;
    if (std::abs(leftmost.x - rightmost.x) < EPS && 
        std::abs(leftmost.y - rightmost.y) < EPS) {
        return {leftmost};
    }
    
    std::vector<Point> hull;
    hull.reserve(points.size());
    
    hull.push_back(leftmost);
    
    quickHullRecursive(points, leftmost, rightmost, 1, hull);
    
    quickHullRecursive(points, rightmost, leftmost, 1, hull);
    
    if (hull.size() > 1 && hull.back() == leftmost) {
        hull.pop_back();
    }
    
    if (hull.size() > 1) {
        std::vector<Point> pruned;
        pruned.reserve(hull.size());
        pruned.push_back(hull[0]);
        
        for (size_t i = 1; i < hull.size(); ++i) {
            const Point& prev = pruned.back();
            const Point& curr = hull[i];
            
            double dx = curr.x - prev.x;
            double dy = curr.y - prev.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist > EPS) {
                pruned.push_back(curr);
            }
        }
        
        hull = pruned;
    }
    
    return hull;
}} // End namespace QuickHull


// --- Main Function (with new output logic) ---
using namespace std;

int main(int argc, char* argv[])
{
    // 1. Check arguments
    if (argc != 2) {
        cerr << "Error: Incorrect usage." << endl;
        cerr << "Usage: " << argv[0] << " <input_filename.txt>" << endl; 
        return 1; 
    }

    // 2. Get the input filename
    string input_path = argv[1]; 

    // --- 3. CONSTRUCT OUTPUT FILENAME ---
    string output_dir = "QUICKHULL_OUTPUT";
    
    // Create the directory
    mkdir(output_dir.c_str(), 0777); 
    
    // Get the base filename from the input path
    string base_filename = input_path;
    size_t last_slash = input_path.rfind('/');
    if (last_slash != string::npos) {
        base_filename = input_path.substr(last_slash + 1);
    }

    // Add "_output" to the base filename
    string modified_base = base_filename;
    size_t dot_pos = modified_base.rfind('.');
    if (dot_pos == string::npos) {
        modified_base += "_output.txt";
    } else {
        modified_base.insert(dot_pos, "_output");
    }

    // Combine directory and new filename
    string output_filename = output_dir + "/" + modified_base;
    // --- End of Modified Logic ---


    // 4. Open the file streams
    ifstream file(input_path);
    ofstream outfile(output_filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_path << endl;
        return 1;
    }
    if (!outfile.is_open()) {
        cerr << "Error: Could not open output file " << output_filename << endl;
        return 1;
    }
    
    // 5. Read ALL points from the file
    std::vector<QuickHull::Point> all_points;
    double x_val, y_val;
    while (file >> x_val >> y_val) {
        all_points.push_back({x_val, y_val});
    }

    if (all_points.empty()) {
        cerr << "Error: Input file " << input_path << " contains no valid points." << endl;
        file.close();
        outfile.close();
        return 1;
    }

    // 6. Run the QuickHull algorithm
    cout << "Processing " << all_points.size() << " points from " << input_path << "..." << endl;
    
    std::vector<QuickHull::Point> hull_points;
    try {
        // Corrected the typo: computeConvexHull
        hull_points = QuickHull::computeConvexHull(all_points);
    } catch (const std::exception& e) {
        cerr << "QuickHull algorithm failed: " << e.what() << endl;
        file.close();
        outfile.close();
        return 1;
    }

    // 7. Write the hull points to the output file
    for(size_t i = 0; i < hull_points.size(); i++)
    {
        outfile << fixed << setprecision(8) << hull_points[i].x << " " << hull_points[i].y << endl;
    }

    // 8. Close files and exit
    file.close();
    outfile.close();
    cout << "Processing complete! Results saved to: " << output_filename << endl;
    
    return 0;
}