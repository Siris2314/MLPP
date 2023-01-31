#ifndef __LAYER__HPP
#define __LAYER__HPP

#include "neuron.hpp"
#include <vector>
#include <stdint.h>

static int layerId = 0;

class Layer{

   public:
    int currLayerSize;
    std::vector<Neuron *> neurons;
    std::vector<double> layerOutput;
    Layer(int, int);

};

#endif