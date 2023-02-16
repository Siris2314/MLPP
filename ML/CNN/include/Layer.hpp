#ifndef _layer_hpp
#define _layer_hpp

#include <vector>
#include "Config.hpp"
#include "../../library/Eigen/Eigen/Core"
#include "RandomNumGen.hpp"
#include "optimizer.hpp"

class Layer
{


    protected:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        const int m_input_size;
        const int m_output_size;

    
    public:
        Layer(const int input_size, const int output_size) :
            m_input_size(input_size),
            m_output_size(output_size)  {}
            
        virtual ~Layer();

        int input_size() const {return m_input_size;}
        int output_size() const {return m_output_size;}

        virtual void init(const Scalar& mean, const Scalar &variance, random_num_gen& rng ) = 0;
        virtual void feedforward(const Matrix& previous_layer_output) = 0;

        virtual const Matrix& output() const = 0;

        virtual void back_prop(const Matrix& previous_layer_output, const Matrix& new_data) = 0;

        virtual const Matrix& back_prop_data()const = 0;

        virtual std::vector<Scalar> get_function_parameters() const = 0;

        virtual void set_params(const std::vector<Scalar>& param) {}

        virtual std::vector<Scalar> get_deriv() const = 0;

};

#endif