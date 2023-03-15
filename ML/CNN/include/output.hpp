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


    public:
        virtual ~Output() { }

        virtual void eval(const Matrix& prev_layer_data, const Matrix& target) = 0;
        virtual void eval(const Matrix& prev_layer_data, const IntVec& target) = 0;

        virtual const Matrix& backprop_data() const = 0;

        virtual Scalar loss() const = 0;

};

#endif