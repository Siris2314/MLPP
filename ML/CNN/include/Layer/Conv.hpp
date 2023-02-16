#ifndef _Conv_hpp
#define _Conv_hpp


#include "../../library/Eigen/Eigen/Core"
#include <vector>
#include <stdexcept>
#include "Config.hpp"
#include "Layer.hpp"
#include "convolution.hpp"
#include "../RandomFunc.hpp"

template<typename Activation>
class Conv : public Layer{

    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Matrix::ConstAlignedMapType ConstAlignedMapMat;
        typedef Vector::ConstAlignedMapType ConstAlignedMapVec;
        typedef Vector::AlignedMapType AlignedMapVec;

        const internal::ConvDims m_dim;

        Vector m_filter_data;
        Vector m_df_data;

        Vector m_bias;
        Vector m_db;

        Matrix m_z;
        Matrix m_a;
        Matrix m_dim;

    public:
        Conv(const int in_width, const int in_height, const int in_channels,const int out_channels, const int window_width, const int window_height):
            Layer(in_width*in_height*in_channels,in_width-window_width+1 * (in_height-window_height+1)*out_channels),
            m_dim(in_channels,out_channels, in_height, in_width, window_height, window_width)

        {}

        void init(const Scalar&mean, const Scalar& variance, random_num_gen& rng){
                const int filter_data_size = m_dim.in_channels*m_dim.out_channels*m_dim.filter_rows*m_dim.filter_cols;
                m_filter_data.resize(filter_data_size);
                m_df_data.resize(filter_data_size);

                m_bias.resize(m_dim.out_channels);
                m_db.resize(m_dim.out_channels);

                internal::set_normal_rand(m_filter_data.data(), filter_data_size, rng, mean, variance);
                internal::set_normal_rand(m_bias.data(), m_dim.out_channels, rng, mean, variance);



        }

        void forward(const Matrix& prev_layer_data){
            const int n = prev_layer_data.cols();
            m_z.resize(this->m_output_size, n);

            internal::convolve_valid(m_dim, prev_layer_data.data(), true, n, m_filter_data.data(),m_z.data());

            int channel_start_row = 0;
            const int channel_new_elem = m_dim.conv_rows*m_dim.conv_cols;

            for(int i = 0; i<m_dim.out_channels; i++, channel_start_row+=channel_new_elem){
                m_z.block(channel_start_row,0, channel_new_elem,n).array() += m_bias[i];
            }

            m_a.resize(this->m_output_size, n);
            Activation::activate(m_z, m_a);

        }   

        const Matrix& output() const{
            return m_a;
        }

        void backpropogate(const Matrix& prev_layer_data, const Matrix& next_layer_data){

        }

        const Matrix& backpropogate_data() const{
            return m_dim;
        }

        void update(Optimizer& opt){
            ConstAlignedMapVec dw(m_df_data.data(), m_df_data.size());
            ConstAlignedMapVec db(m_db.data(), m_db.size());
            AlignedMapVec w(m_filter_data.data(), m_df_data.size());
            AlignedMapVec b(m_db.data(), m_bias.size());

            opt.update(dw, w);
            opt.update(db, b);
        }

};


#endif