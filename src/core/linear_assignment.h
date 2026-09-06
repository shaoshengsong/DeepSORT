#pragma once
#include <vector>
#include <utility>
#include "lap_solver.h"

namespace dt {

// Thresholded minimum-cost linear assignment over cost (rows x cols, float).
// Mirrors Python's `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`:
//   only edges with cost <= thresh may match; returns the optimal
//   (maximum cardinality, minimum total cost) assignment.
// Implementation: pad to an N x N square matrix; feasible edges keep their cost,
//   infeasible edges become LARGE, dummy edges get cost = thresh, then
//   lapjv_solve finds the minimum-cost perfect matching and we keep the
//   real pairs with cost <= thresh.
void linear_assignment(const std::vector<std::vector<float>>& cost,
                       float thresh,
                       std::vector<std::pair<int, int>>& matches,
                       std::vector<int>& unmatched_a,
                       std::vector<int>& unmatched_b);

} // namespace dt
