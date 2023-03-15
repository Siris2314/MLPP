#ifndef CrossEntropy_HPP
#define CrossEntropy_HPP

#include <Eigen/Core>
#include "output.hpp"
#include "../Config.hpp"
class CrossEntropy: public Output{

private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    Matrix m_din;

public:
    void eval(const Matrix& prev_layer_data, const Matrix& target){

        const int n = prev_layer_data.cols();
        const int n_classes = prev_layer_data.rows();
        m_din.resize(n_classes, n);
        m_din.noalias()= -target.cwiseQuotient(prev_layer_data);
    }



    void eval(const Matrix& prev_layer_data, const IntVec& target){
        const int n = prev_layer_data.cols();
        const int n_classes = prev_layer_data.rows();
        m_din.resize(n_classes, n);
        m_din.setZero();
        for(int i=0; i<n; i++){
            m_din(target[i], i) = -Scalar(1)/prev_layer_data(target[i], i);
        }
    }


    const Matrix& backprop_data() const{
        return m_din;
    }

    Scalar loss() const{
        Scalar result = Scalar(0);
        const int n = m_din.size();
        const Scalar* din_data = m_din.data();
        for(int i=0; i<n; i++){
            if(din_data[i] < 0){
                result += std::log(-din_data[i]);
            }
        }

        return result/m_din.cols();
    }
};






#endif