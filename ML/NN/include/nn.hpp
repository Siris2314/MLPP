#ifndef __NN__HPP
#define __NN__HPP

#include "../../include/data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "../../include/common.hpp"

class Network : public common_data
{

    public:
        std::vector<Layer *> layers;
        double lR; //Learning Rate
        double testingAccuracy;
        Network(std::vector<int> spec, int, int, double);
        ~Network();
        std::vector<double> feedForward(data *data);
        double activationFunction(std::vector<double>, std::vector<double>);
        double transfer(double);
        double transferDerivative(double);
        void backPropagate(data *data);
        void updateWeights(data *data);
        int predict(data *data);
        void train(int);
        double test();
        void validate();

};


#endif