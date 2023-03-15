#ifndef AdaGrad_hpp
#define AdaGrad_hpp


#include <Eigen/Core>
#include "../Config.hpp"
#include "./optimizer.hpp"
#include "./sparsepp.hpp"

class AdaGrad : public Optimizer {

    private:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

        spp::sparse_hash_map<const Scalar*, Array> m_history;


    public:
        Scalar l_rate;
        Scalar eps;

        AdaGrad():
            l_rate(Scalar(0.001)),
            eps(Scalar(1e-7))
        {}

        void reset(){
            m_history.clear();
        }

        void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec){
            Array& grad = m_history[vec.data()];

            if(grad.size() == 0){
                grad.resize(dvec.size());
                grad.setZero();
            }

            grad += dvec.array().square();

            vec.array() -= l_rate * dvec.array() / (grad.sqrt() + eps);
        }

};





#endif