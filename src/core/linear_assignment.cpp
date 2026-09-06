#include "linear_assignment.h"
#include <algorithm>
#include <cmath>

namespace dt {

void linear_assignment(const std::vector<std::vector<float>>& cost,
                       float thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& unmatched_a,
                       std::vector<int>& unmatched_b) {
    matches.clear();
    unmatched_a.clear();
    unmatched_b.clear();

    int rows = static_cast<int>(cost.size());
    int cols = rows ? static_cast<int>(cost[0].size()) : 0;

    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) unmatched_a.push_back(i);
        for (int j = 0; j < cols; ++j) unmatched_b.push_back(j);
        return;
    }

    const int n = std::max(rows, cols);
    cost_t** C = new cost_t*[n];
    for (int i = 0; i < n; ++i) C[i] = new cost_t[n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < rows && j < cols) {
                float c = cost[i][j];
                // Infeasible edges (cost above threshold) become LARGE so they are never picked.
                C[i][j] = (c <= thresh) ? static_cast<cost_t>(c) : static_cast<cost_t>(LARGE);
            } else {
                // Dummy edges (for padding to a square matrix) get cost = threshold, so unmatched rows/cols prefer them.
                C[i][j] = static_cast<cost_t>(thresh);
            }
        }
    }

    int_t* x = new int_t[n];
    int_t* y = new int_t[n];
    lapjv_solve(static_cast<uint_t>(n), C, x, y);

    for (int i = 0; i < rows; ++i) {
        int j = x[i];
        if (j >= 0 && j < cols && cost[i][j] <= thresh) {
            matches.push_back({i, j});
        } else {
            unmatched_a.push_back(i);
        }
    }
    for (int j = 0; j < cols; ++j) {
        int i = y[j];
        if (!(i >= 0 && i < rows && cost[i][j] <= thresh)) {
            unmatched_b.push_back(j);
        }
    }

    for (int i = 0; i < n; ++i) delete[] C[i];
    delete[] C;
    delete[] x;
    delete[] y;
}

} // namespace dt
