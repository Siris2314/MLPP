#ifndef Output_hpp
#define Output_hpp

#include <Eigen/Core>
#include <stdexcept>
#include "Config.hpp"

class Output{
    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::RowVectorXi IntVec;

};

#endif