#pragma once

#include "../utils/publicBeliefState.h"
#include "../utils/utils.h"
#include "../settings.h"
#include "../netHolder.h"
#include "net.h"
#include "dataSaver.h"
#include "netHelper.h"

#include <torch/torch.h>
#include <memory>
#include <chrono>

class Inference {

public:
    Inference() = default;
    torch::Tensor evaluate(std::shared_ptr<PublicBeliefState> node, torch::Tensor player1Range, torch::Tensor player2Range, std::optional<int> modelIndex = std::nullopt);

private:

};