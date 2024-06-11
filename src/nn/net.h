#pragma once

#include "netHelper.h"

#include <torch/torch.h>

class NetImpl : public torch::nn::Module {

public:
    NetImpl(int input_size_, int hidden_size_, int output_size_);

    torch::Tensor forward(torch::Tensor x);

private:
    int input_size;
    int hidden_size;
    int output_size;
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::Linear fc4{nullptr};
    // torch::nn::BatchNorm1d bn1{nullptr}; // BatchNorm for the first hidden layer
    // torch::nn::BatchNorm1d bn2{nullptr}; // BatchNorm for the second hidden layer
    // torch::nn::BatchNorm1d bn3{nullptr}; // BatchNorm for the third hidden layer
    torch::nn::LayerNorm ln1{nullptr}; // LayerNorm for the first hidden layer
    torch::nn::LayerNorm ln2{nullptr}; // LayerNorm for the second hidden layer
    torch::nn::LayerNorm ln3{nullptr}; // LayerNorm for the third hidden layer

    // torch::nn::Dropout drop2{nullptr};  // Dropout layer after the second hidden layer
    // torch::nn::Dropout drop3{nullptr};  // Dropout layer after the third hidden layer

};
TORCH_MODULE(Net);