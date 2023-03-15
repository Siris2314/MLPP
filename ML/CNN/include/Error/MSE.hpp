#ifndef MSE_hpp
#define MSE_hpp

#include <Eigen/Core>
#include "../Config.hpp"
#include "output.hpp"

class MSE : public Output {

    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Matrix m_din;

    public:
        void eval(const Matrix& prev_layer_data, const Matrix& target){
            const int n = prev_layer_data.cols();
            const int n_classes = prev_layer_data.rows();


            m_din.resize(n_classes, n);
            m_din.noalias() = prev_layer_data - target;
            
        }

        const Matrix& backprop_data() const{
            return m_din;
        }

        Scalar loss() const{
            return m_din.squaredNorm() / m_din.cols()*0.5;
        }
};

#endif