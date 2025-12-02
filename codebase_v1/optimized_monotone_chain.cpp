/*
 * Optimized Monotone Chain Convex Hull Algorithm
 * 
 * Enhanced version of Andrew's algorithm with O(n) sorted check.
 * Achieves best-case O(n) complexity for pre-sorted data.
 * 
 * Time Complexity:
 * - Best: O(n) for sorted input
 * - Worst: O(n log n) for unsorted input
 * 
 * Usage: ./optimized_monotone_chain <input_file>
 */

#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h> // For mkdir

using namespace std;

/**
 * Point structure for 2D coordinates
 */
struct pt {
    double x, y;
};

// --- Orientation Functions ---

/**
 * Compute orientation of ordered triplet (a, b, c)
 */
int orientation(pt a, pt b, pt c) {
    double v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y);
    if (v < 0) return -1; // clockwise
    if (v > 0) return +1; // counter-clockwise
    return 0; // collinear
}

/**
 * Check if three points make a clockwise turn
 */
bool cw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}

/**
 * Check if three points make a counterclockwise turn
 */
bool ccw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o > 0 || (include_collinear && o == 0);
}

/**
 * Optimized Monotone Chain with sorted check
 * 
 * Key optimization: O(n) check before O(n log n) sort.
 * Skips sorting for pre-sorted data.
 */
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.size() <= 2)
        return;

    // Define the comparison lambda
    auto lexicographical_compare = [](pt a, pt b) {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    };

    // OPTIMIZATION: Check if already sorted (O(n))
    if (!is_sorted(a.begin(), a.end(), lexicographical_compare)) {
        // Run the O(n log n) sort only if needed
        sort(a.begin(), a.end(), lexicographical_compare);
    }

    pt p1 = a[0], p2 = a.back();
    vector<pt> up, down;
    up.push_back(p1);
    down.push_back(p1);
    for (int i = 1; i < (int)a.size(); i++) {
        if (i == a.size() - 1 || cw(p1, a[i], p2, include_collinear)) {
            while (up.size() >= 2 && !cw(up[up.size()-2], up[up.size()-1], a[i], include_collinear))
                up.pop_back();
            up.push_back(a[i]);
        }
        if (i == a.size() - 1 || ccw(p1, a[i], p2, include_collinear)) {
            while (down.size() >= 2 && !ccw(down[down.size()-2], down[down.size()-1], a[i], include_collinear))
                down.pop_back();
            down.push_back(a[i]);
        }
    }

    if (include_collinear && up.size() == a.size()) {
        reverse(a.begin(), a.end());
        return;
    }
    a.clear();
    for (int i = 0; i < (int)up.size(); i++)
        a.push_back(up[i]);
    for (int i = down.size() - 2; i > 0; i--)
        a.push_back(down[i]);
}

// --- Solve Function ---
void solve(vector<pt>& a, ofstream &outfile, int p)
{
    // The vector 'a' already contains all points.
    
    // Process the convex hull
    convex_hull(a, 0); 
    
    // Output the results
    for(size_t i = 0; i < a.size(); i++)
    {
        outfile << fixed << setprecision(8) << a[i].x << " " << a[i].y << endl;
    }
}

// --- Main Function (with new output logic) ---
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
    string output_dir = "OPTIMIZED_MONOTONE_OUTPUT";
    
    // Create the directory (0777 are permissions).
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
    ifstream file(input_path); // Use original input_path
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