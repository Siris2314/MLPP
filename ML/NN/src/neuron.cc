#include "neuron.hpp"
#include <random>


double randNumGenerator(double min, double max){
    double random = (double) rand() / RAND_MAX;
    return min + random * (max - min);
}


Neuron::Neuron(int prevLayerSize, int currLayerSize){
    initWeights(prevLayerSize);
}

void Neuron::initWeights(int prevLayerSize){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for(int i = 0; i < prevLayerSize + 1; i++){
        weights.push_back(randNumGenerator(-1.0, 1.0));
    }

}