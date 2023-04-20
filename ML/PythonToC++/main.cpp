#include <iostream>
#include <torch/script.h>

int main(){
    torch::jit::script::Module net = torch::jit::load("../models/net.pt");
    torch:: Tensor x = torch::randn({1,100});
    std::vector<torch::jit::IValue> input;
    input.push_back(x);
    for(int i = 0; i<10; i++){
        auto out = net.forward(input);
        std::cout <<out <<std::endl;
    }

}