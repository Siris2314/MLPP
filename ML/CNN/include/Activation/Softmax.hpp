#ifndef Softmax_HPP
#define Softmax_HPP


#include <Eigen/Core>
#include "../Config.hpp"


class Softmax
{
    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:
        static inline void activate(const Matrix& Z, Matrix& A){
            A.noalias() = Z;
        }

        static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G){
            G.noalias() = F;
        }
};

#endif