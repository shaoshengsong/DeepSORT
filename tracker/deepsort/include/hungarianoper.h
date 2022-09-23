#ifndef HUNGARIANOPER_H
#define HUNGARIANOPER_H
#include "munkres.h"
#include "dataType.h"

class HungarianOper {
public:
    static Eigen::Matrix<float, -1, 2, Eigen::RowMajor> Solve(const DYNAMICM &cost_matrix);
};

#endif // HUNGARIANOPER_H
