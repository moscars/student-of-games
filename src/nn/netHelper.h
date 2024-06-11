#pragma once

#include "../settings.h"
#include "../utils/card.h"
#include "../utils/utils.h"

#include <torch/torch.h>

namespace netHelper{
    torch::Tensor getBoardMask(torch::Tensor boardTen, int bitBoard);
    torch::Tensor customGELU(torch::Tensor x);

    extern std::unordered_map<int, torch::Tensor> maskCache;
    extern std::shared_mutex maskCacheMutex;
};