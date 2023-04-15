#pragma once

#include <iostream>
#include <torch/torch.h>

struct NetworkImpl : torch::nn::Module{
   
    NetworkImpl(int dims1_dim, int dims2_dim):
        dims1(dims1_dim, dims1_dim), dims2(dims2_dim, dims2_dim), out(dims2_dim, 1){
        register_module("dims1", dims1);
        register_module("dims2", dims2);
        register_module("out", out);
        }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(dims1(x));
        x = torch::relu(dims2(x));
        x = out(x);
        return x;
    }

    torch::nn::Linear dims1{nullptr}, dims2{nullptr}, out{nullptr};
};

TORCH_MODULE(Network);


