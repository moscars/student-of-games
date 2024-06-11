#pragma once

#include "../utils/utils.h"
#include "../utils/card.h"
#include "../settings.h"
#include "../solving/deepResolving.h"
#include "net.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <sstream>
#include <future>

/*
    Used to solve situations that the neural network was tasked with during training.
    This solved situations can then be used as targets for the neural network.
*/

class SituationSolver{

public:
    SituationSolver(std::shared_ptr<Net> model_, const std::string & filename_);
    void solve();

private:
    std::shared_ptr<Net> model;
    std::string filename;
    int numCores;

    std::vector<torch::Tensor> solveLines(const std::vector<std::string> & lines);
};
