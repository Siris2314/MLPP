#ifndef __Identity_HPP
#define _Identity_HPP

#include <Eigen/Core>
#include "../Config.hpp"

class Identity
{
    private: 
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;


    public:
        static inline void activate(const Matrix& z, Matrix& A){
            A.noalias() = z;

        }

        static inline void apply_jacobian(const Matrix& z, const Matrix& A, const Matrix& F, Matrix& G){
            G.noalias() = F;
        }


};




#endif