#ifndef SGD_HPP
#define SGD_HPP



#include <Eigen/Core>
#include "../Config.hpp"
#include "../optimizer.hpp"

class SGD : public Optimizer{
    public:
        Scalar m_learn_rate;
        Scalar m_decay_rate;

        SGD():
            m_learn_rate(Scalar(0.001)),
            m_decay_rate(Scalar(0.0))
        {}

        void update(ConstAlignedMapVec& dvec, AlignedMapVec& vec){
            vec.noalias() -= m_learn_rate *(dvec + m_decay_rate * vec);
        }
}


#endif