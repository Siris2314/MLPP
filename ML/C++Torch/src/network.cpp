#include "../include/network.h"
#include <iostream>
#include <torch/torch.h>

int main(){
    Network network(50,10);
    std::cout << network << std::endl;
}