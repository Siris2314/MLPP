#ifndef _Pooling_hpp
#define _Pooling_hpp

#include "../../../library/Eigen/Eigen/Core"
#include <vector>
#include <stdexcept>
#include "../Layer.hpp"
#include "../Config.hpp"
#include "maxaverage.hpp"


template<typename ActivationFunction>
class Pooling : public Layer
{

private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::MatrixXi IntMatrix;

    const int m_channel_rows;
    const int m_channel_cols;
    const int m_in_channel;
    const int m_pool_rows;
    const int m_pool_cols;
    const int m_out_rows;
    const int m_out_cols;


    IntMatrix m_loc;
    Matrix m_z;
    Matrix m_a;
    Matrix m_din;


    public:
        Pooling(const int in_width, const int in_height, const int in_channel, const int pool_width, const int pool_height):
            Layer(in_width*in_height*in_channel, (in_width/pool_width)*(in_height/pool_height)*in_channel),
            m_channel_rows(in_height),
            m_channel_cols(in_width),
            m_in_channel(in_channel),
            m_pool_rows(pool_height),
            m_pool_cols(pool_width),
            m_out_rows(m_channel_rows/m_pool_rows),
            m_out_cols(m_channel_cols/m_pool_cols)
        {}

        void init(const Scalar& mean, const Scalar& variance, RNG& rng){}

        void feedforward(const Matrix& prev_layer_data){
            const int n = prev_layer_data.cols();
            m_loc.resize(this->m_output_size, n);
            m_z.resize(this->m_output_size, n);

            int* loc_data = m_loc.data();
            const int channel_final = prev_layer_data.size();
            const int channel_stride = m_channel_rows*m_channel_cols;
            const int col_end_gap = m_channel_rows*(m_pool_cols)*m_out_cols;
            const int col_stride = m_channel_rows*m_pool_cols;
            const int row_end_gap = m_channel_rows*(m_pool_rows);
            for(int i=0; i < channel_end; i+=channel_stride){
                const int col_end = i + col_end_gap;
                for(int j=i; j<col_end; j+=col_stride){
                    const int row_end = j+ row_end_gap;
                    for(int k = j; k<row_end; k+=m_pool_rows, loc_data++){
                        loc_data = k;
                    }
                }
            }

            loc_data = m_loc.data();
            const int* const loc_end = loc_data + m_loc.size();
            Scalar* z_data = m_z.data(); 
            const Scalar* src=prev_layer_data.data();
            for(; loc_data<loc_end; loc_data++, z_data++){
                const int offser = *loc_data;
                *z_data = internal::find_block_max(src+offset, m_pool_rows, m_pool_cols, m_channel_rows, *loc_data);
                *loc_data += offset;
            }

            m_a.resize(this->m_output_size, n);

            ActivationFunction::activate(m_z, m_a);
        }

        const Matrix& output() const {
            return m_a;
        }

        void backprop(const Matrix& prev_layer_data, const Matrix& next_layer_data){
    
        }

        const Matrix& backprop_data() const {
            return m_din;
        }

        void update(Optimzer& opt){
            
        }
};

#endif