#pragma once

#include "../utils/card.h"
#include "../utils/utils.h"
#include "../settings.h"
#include "handComp.h"

#include <vector>
#include <cassert>

#include <torch/torch.h>

namespace equity{
    // When we call all in we get the equity as the sum of our opponents range when we win minus the sum of the opponents range
    // when we lose
    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCall(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2);
    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCallNoBoard(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2);
    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCallOneBoard(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2);
    // The equity of getting the other player to fold for each possible hand
    // is the sum of the the opponents valid range (since they folded we get the full equity)
    // And the mirror of this is that the equity lost when we fold is the sum of our valid range
    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityFold(const std::vector<Card> & board, torch::Tensor opponentRange);
};