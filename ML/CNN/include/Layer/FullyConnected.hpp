#ifndef _FullyConnected_hpp
#define _FullyConnected_hpp


#include "../../../library/Eigen/Eigen/Core"
#include <vector>
#include <stdexcept>
#include "../Config.hpp"
#include "../Layer.hpp"
#include "../RandomNumGen.hpp"
#include "../RandomFunc.hpp"



template<typename Activation_Function>  
class FullyConnected : public Layer
{

  private:
    typdef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typdef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Matrix m_weight;
    Vector m_bias;
    Matrix m_dw;
    Vector m_db;
    Matrix m_z;
    Matrix m_a;
    Matrix m_dim;

    public:
        FullyConnected(const int input_size, const int output_size) :
            Layer(input_size, output_size)
        {}

        void init(const Scalar& mean, const Scalar &variance, random_num_gen& rng){
            m_weight.resize(this->m_input_size, this->m_output_size);
            m_bias.resize(this->m_output_size);
            m_dw.resize(this->m_input_size, this->m_output_size);
            m_db.resize(this->m_output_size);

            internal::set_normal_rand(m_weight.data(), m_weight.size(), rng, mean, variance);
            internal::set_normal_rand(m_bias.data(), m_bias.size(), rng, mean, variance);
        }

        void feedforward(const Matrix& previous_layer_output){
            const int n = previous_layer_output.cols();
            m_z.resize(this->m_output_size, n);
            m_z.noalias() = m_weight.transpose() * previous_layer_output;
            m_z.colwise() += m_bias;

            m_a.resize(this->m_output_size, n);
            Activation_Function::activate(m_z, m_a);
        }

        const Matrix& output() const{
            return m_a;
        }

        void back_prop(const Matrix& previous_layer_output, const Matrix& new_data){
            const int n = previous_layer_output.cols();
            m_dim.resize(this->m_output_size, n);
            Activation_Function::derivative(m_z, m_dim);
            m_dim.array() *= new_data.array();

            m_dw.noalias() = previous_layer_output * m_dim.transpose();
            m_db = m_dim.rowwise().sum();

            m_dim.resize(this->m_input_size, n);
            m_dim.noalias() = m_weight * m_dim;
        }

        const Matrix& back_prop_data() const{
            return m_dim;
        }

        void update(Optimizer& opt){
            ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
            ConstAlignedMapVec db(m_db.data(), m_db.size());
            AlignedMapVec w(m_weight.data(), m_weight.size());
            AlignedMapVec b(m_bias.data(), m_bias.size());

            opt.update(dw, w);
        }

        std::vector<Scalar> get_function_parameters() const{
            
        }

        void set_params(const std::vector<Scalar>& param){


        }

        std::vector<Scalar> get_derivatives() const {}

        


};



#endif