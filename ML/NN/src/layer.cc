#include "layer.hpp"

Layer::Layer(int prevLayerSize, int currLayerSize){
    for(int i = 0; i<currLayerSize; i++){
        neurons.push_back(new Neuron(prevLayerSize, currLayerSize));
    }
    this->currLayerSize = currLayerSize;
}