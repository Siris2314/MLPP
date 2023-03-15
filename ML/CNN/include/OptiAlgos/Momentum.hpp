#ifndef Momentum_HPP
#define Momentum_HPP

#include <Eigen/Core>
#include "../Config.hpp"
#include "../sparsepp.hpp"
#include "../optimizer.hpp"



class Momentum : public Optimizer {

    private:
        typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
        spp::sparse_hash_map<const Scalar*, Array> m_history;

    public:
        Scalar l_rate;
        Scalar momentum_rate;
        Scalar decay_rate;

        Momentum():
            l_rate(Scalar(0.001)),
            momentum_rate(Scalar(0.9)),
            decay_rate(Scalar(0))
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

            grad = grad * momentum_rate + l_rate * (dvec + decay_rate * vec).array();

            vec.array() -= grad;
        }

};





#endif