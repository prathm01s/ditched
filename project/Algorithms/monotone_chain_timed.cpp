#include<bits/stdc++.h>
#include<fstream>
#include<string>
#include <sys/stat.h>
#include <chrono>
#include <sys/resource.h>

using namespace std;
struct pt { double x, y; };
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
bool ccw(pt a, pt b, pt c, bool include_collinear) {
    int o = orientation(a, b, c);
    return o > 0 || (include_collinear && o == 0);
}
void convex_hull(vector<pt>& a, bool include_collinear = false) {
    if (a.size() <= 2) return;
    sort(a.begin(), a.end(), [](pt a, pt b) {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    });
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
    for (int i = 0; i < (int)up.size(); i++) a.push_back(up[i]);
    for (int i = down.size() - 2; i > 0; i--) a.push_back(down[i]);
}

int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    ifstream file(argv[1]);
    if (!file.is_open()) return 1;

    vector<pt> points;
    double x, y;
    string line;
    // Skip Header
    while (file.peek() != EOF) {
        char c = file.peek();
        if (isdigit(c) || c == '-' || c == '.') break;
        getline(file, line);
    }
    // Read Data with Comma Handling
    while (file >> x) {
        if (file.peek() == ',') file.ignore(); 
        if (file >> y) points.push_back({x, y});
    }
    file.close();

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