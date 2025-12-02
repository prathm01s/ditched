/*
 * Graham Scan Convex Hull Algorithm
 * 
 * This implementation computes the convex hull of a set of 2D points using
 * Graham's scan algorithm, a classic approach with O(n log n) time complexity.
 * 
 * Algorithm Overview:
 * 1. Find the lowest point (smallest y, then smallest x) as the pivot
 * 2. Sort all other points by polar angle with respect to the pivot
 * 3. Use a stack to build the hull by checking turns (left/right)
 * 4. Output hull vertices in counterclockwise order
 * 
 * Time Complexity: O(n log n) - dominated by sorting
 * Space Complexity: O(n) - for storing the hull
 * 
 * Usage: ./graham_scan <input_file> <output_file> <print_flag>
 */

#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h> // For mkdir function

using namespace std;

/**
 * Point structure representing 2D coordinates
 */
struct pt {
    double x, y;
    bool operator == (pt const& t) const {
        return x == t.x && y == t.y;
    }
};

// --- Graham Scan Algorithm Functions ---

/**
 * Compute orientation of ordered triplet (a, b, c)
 * 
 * Uses the cross product formula:
 * orientation = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
 * 
 * Returns:
 *  -1 : Clockwise orientation
 *   0 : Collinear points
 *  +1 : Counterclockwise orientation
 */
int orientation(pt a, pt b, pt c) {
    double v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y);
    if (v < 0) return -1; // clockwise
    if (v > 0) return +1; // counter-clockwise
    return 0;
}

/**
 * Check if three points make a clockwise turn (or are collinear if flag set)
 */
bool cw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}

/**
 * Check if three points are collinear
 */
bool collinear(pt a, pt b, pt c) { return orientation(a, b, c) == 0; }

/**
 * Graham Scan convex hull algorithm
 * 
 * Modifies the input vector 'a' to contain only the hull vertices.
 * 
 * Steps:
 * 1. Find pivot point p0 (lowest y-coordinate, leftmost if tied)
 * 2. Sort points by polar angle relative to p0
 * 3. Build hull using stack-based scan
 * 
 * @param a Vector of points (modified in-place to contain hull)
 * @param include_collinear Whether to include collinear points on hull boundary
 */
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    // Find the pivot point (lowest, leftmost)
    pt p0 = *min_element(a.begin(), a.end(), [](pt a, pt b) {
        return make_pair(a.y, a.x) < make_pair(b.y, b.x);
    });
    
    // Sort by polar angle with respect to p0
    sort(a.begin(), a.end(), [&p0](const pt& a, const pt& b) {
        int o = orientation(p0, a, b);
        if (o == 0)
            // If collinear, sort by distance from p0
            return (p0.x-a.x)*(p0.x-a.x) + (p0.y-a.y)*(p0.y-a.y)
                < (p0.x-b.x)*(p0.x-b.x) + (p0.y-b.y)*(p0.y-b.y);
        return o < 0;
    });
    
    // Handle collinear points on the far side
    if (include_collinear) {
        int i = (int)a.size()-1;
        while (i >= 0 && collinear(p0, a[i], a.back())) i--;
        reverse(a.begin()+i+1, a.end());
    }

    // Build hull using stack
    vector<pt> st;
    for (int i = 0; i < (int)a.size(); i++) {
        // Remove points that make a right turn
        while (st.size() > 1 && !cw(st[st.size()-2], st.back(), a[i], include_collinear))
            st.pop_back();
        st.push_back(a[i]);
    }

    // Handle degenerate case of single unique point
    if (include_collinear == false && st.size() == 2 && st[0] == st[1])
        st.pop_back();

    a = st;
}

/**
 * Solve function: computes hull and writes output
 */
void solve(vector<pt>& a, ofstream &outfile, int p)
{
    // The vector 'a' already contains all points.
    
    // Process the convex hull
    convex_hull(a, 0); 
    
    // Output the results WITHOUT "CASE 1"
    for(size_t i = 0; i < a.size(); i++)
    {
        // Use fixed and setprecision for double coordinates
        outfile << fixed << setprecision(8) << a[i].x << " " << a[i].y << endl;
    }
}

// --- !!! MODIFIED main function !!! ---
int main(int argc, char* argv[])
{
    // 1. Check for the correct number of command-line arguments
    if (argc != 2) {
        // Print error message to cerr (standard error)
        cerr << "Error: Incorrect usage." << endl;
        // argv[0] is the name of the program itself
        cerr << "Usage: " << argv[0] << " <input_filename.txt>" << endl; 
        return 1; // Return an error code
    }

    // 2. Get the input filename
    string input_path = argv[1]; 

    // --- 3. CONSTRUCT OUTPUT FILENAME (Modified Logic) ---
    string output_dir = "GRAHAM_SCAN_OUTPUT"; // <-- Unique folder
    
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
    vector<pt> all_points;
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

    // 6. Run the solve function once
    cout << "Processing " << all_points.size() << " points from " << input_path << "..." << endl;
    solve(all_points, outfile, 1); 

    // 7. Close files and exit
    file.close();
    outfile.close();
    cout << "Processing complete! Results saved to: " << output_filename << endl;
    
    return 0;
}