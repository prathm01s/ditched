#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;

// --- ALGORITHM UNCHANGED ---
struct pt {
    double x, y;
    bool operator == (pt const& t) const { return x == t.x && y == t.y; }
};
int orientation(pt a, pt b, pt c) {
    double v = a.x*(b.y-c.y)+b.x*(c.y-a.y)+c.x*(a.y-b.y);
    if (v < 0) return -1;
    if (v > 0) return +1;
    return 0;
}
bool cw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o < 0 || (include_collinear && o == 0);
}
bool collinear(pt a, pt b, pt c) { return orientation(a, b, c) == 0; }
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.empty()) return; // Safety check
    pt p0 = *min_element(a.begin(), a.end(), [](pt a, pt b) {
        return make_pair(a.y, a.x) < make_pair(b.y, b.x);
    });
    sort(a.begin(), a.end(), [&p0](const pt& a, const pt& b) {
        int o = orientation(p0, a, b);
        if (o == 0)
            return (p0.x-a.x)*(p0.x-a.x) + (p0.y-a.y)*(p0.y-a.y)
                < (p0.x-b.x)*(p0.x-b.x) + (p0.y-b.y)*(p0.y-b.y);
        return o < 0;
    });
    if (include_collinear) {
        int i = (int)a.size()-1;
        while (i >= 0 && collinear(p0, a[i], a.back())) i--;
        reverse(a.begin()+i+1, a.end());
    }
    vector<pt> st;
    for (int i = 0; i < (int)a.size(); i++) {
        while (st.size() > 1 && !cw(st[st.size()-2], st.back(), a[i], include_collinear))
            st.pop_back();
        st.push_back(a[i]);
    }
    if (include_collinear == false && st.size() == 2 && st[0] == st[1])
        st.pop_back();
    a = st;
}

// --- ROBUST MAIN FUNCTION ---
int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    ifstream file(argv[1]);
    if (!file.is_open()) return 1;

    vector<pt> points;
    double x, y;
    string line;
    
    // 1. SKIP HEADER (Skip lines starting with non-digit/minus)
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line); // Consume header line
    }

    // 2. READ DATA (Handle '123.45,678.90' format)
    while (file >> x) {
        char c;
        // Eat potential comma
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) {
            points.push_back({x, y});
        }
    }
    file.close();

    if (points.size() < 3) {
        // Return 0 time if not enough points to hull
        cerr << "BENCHMARK_TIME_SEC=0.0" << endl;
        cerr << "BENCHMARK_MEM_KB=0" << endl;
        return 0;
    }

    struct rusage usage_end;
    auto start_time = chrono::high_resolution_clock::now();

    convex_hull(points, false);

    auto end_time = chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_end);

    chrono::duration<double> elapsed = end_time - start_time;
    cerr << "BENCHMARK_TIME_SEC=" << fixed << setprecision(9) << elapsed.count() << endl;
    cerr << "BENCHMARK_MEM_KB=" << usage_end.ru_maxrss << endl;
    return 0;
}