#ifndef Softmax_HPP
#define Softmax_HPP


#include <Eigen/Core>
#include "../Config.hpp"


class Softmax
{
    private:
        typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArr;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    public:
        static inline void activate(const Matrix& Z, Matrix& A){
            A.array()=(Z.array().rowwise() - Z.array().rowwise().maxCoeff()).exp();
            RowArr colsum  = A.array().colwise().sum();
            A.array().rowwise() /= colsum;
        }

        static inline void apply_jacobian(const Matrix& Z, const Matrix& A, const Matrix& F, Matrix& G){
            RowArr a_dot_f = A.cwiseProduct(F).rowwise().sum();
            G.array() = A.array()*(F.array().rowwise() - a_dot_f);
        }
};

#endif