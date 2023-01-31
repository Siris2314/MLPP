#ifndef __NEURON__HPP
#define __NEURON__HPP

#include <cmath>
#include <vector>
#include <stdio.h>

class Neuron
{
  public:
    double final;
    double delta;
    std::vector<double> weights;
    Neuron(int, int);
    void initWeights(int);

};

#endif