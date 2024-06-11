#pragma once

#include "../settings.h"

#include <string>
#include <cassert>
#include <fstream>

#include <torch/torch.h>

class DataGenerator {

public:
    DataGenerator() = default;
    void generate(const std::string & filename_);

private:
    void generateRange(std::vector<float> & range, float mass, size_t left, size_t right);
    void saveData(const std::vector<float> & board, const std::vector<float> & actingPlayer, 
                                                                    float betP1, 
                                                                    float betP2, 
                                                                    const std::vector<float> & rangeP1, 
                                                                    const std::vector<float> & rangeP2, 
                                                                    const std::string & filename);

};