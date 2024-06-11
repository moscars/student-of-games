#pragma once

#include "../utils/utils.h"
#include "../settings.h"
#include "net.h"

#include <map>
#include <torch/torch.h>
#include <string>
#include <fstream>

class Trainer {

public:
    Trainer(std::vector<std::string> playerFiles_, bool loadModel_, int num_epochs_);
    void train(std::shared_ptr<Net> model, float lr, int currentName);
    void getLoss(std::shared_ptr<Net> model, const std::vector<std::pair<torch::Tensor, torch::Tensor>> & validation);

private:
    // Read training data from files
    // And returns two tensors
    // One as the features for the network
    // And the other as the target values
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::pair<torch::Tensor, torch::Tensor>>> read_training_data();

    std::vector<std::string> playerFiles;
    bool loadModel;
    int num_epochs;
};