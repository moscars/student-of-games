#pragma once

#include "../utils/publicBeliefState.h"
#include "../game/equity.h"
#include "../settings.h"
#include "../nn/inference.h"

#include <memory>

class Resolving;

class PBSCFR {

public:
    PBSCFR() : regret_epsilon(1.0f/1000000000.0f) {}

    void runCFR(std::shared_ptr<PublicBeliefState> tree);

    // used in resolving
    void cfrIter(std::shared_ptr<PublicBeliefState> node, torch::Tensor opponentRange, int iter);
    void normalizeStrat(std::shared_ptr<PublicBeliefState> node);

private:
    void cfrDFS(std::shared_ptr<PublicBeliefState> node, int iter);
    void updateAverageStrategy(std::shared_ptr<PublicBeliefState> node, torch::Tensor current_strategy, int iter);

    float regret_epsilon;

    // Inference inference;
};