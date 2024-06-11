#pragma once

#include "../utils/card.h"
#include "equity.h"

#include <vector>
#include <unordered_map>

#include <torch/torch.h>

class EquityCache {

public:
    EquityCache() = default;

    torch::Tensor getTerminalEquityCall(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2);
    torch::Tensor getTerminalEquityFold(const std::vector<Card> & board, torch::Tensor opponentRange);

private:
    std::unordered_map<int, torch::Tensor> cache;
    std::unordered_map<int, torch::Tensor> foldCache;

};