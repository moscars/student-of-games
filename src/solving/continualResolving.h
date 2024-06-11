#pragma once

#include "../utils/publicBeliefState.h"
#include "../settings.h"
#include "../game/gameState.h"
#include "deepResolving.h"

#include <memory>
#include <cassert>

#include <torch/torch.h>

class ContinualResolving {

public:
    ContinualResolving();

    GameState act(GameState state);
    GameState actUniformRandom(GameState state);

    torch::Tensor getPlayerRange() const;
    int getPosition() const;
    void setModelEvaluation(bool modelEvaluation_);
    float getAIVatFromBoardAction(size_t actionIdx, const std::vector<Card> & board);
    torch::Tensor getOpponentCFVBoundBoard(const std::vector<Card> & board, bool adjustRange);

    torch::Tensor getPrevPlayerRange() const;
    torch::Tensor getPrevPlayerStrategy() const;
    float findAIVATValue(size_t actionIdx, torch::Tensor p1Range, torch::Tensor p2Range, int modelToUse);

private:
    size_t sampleAction(torch::Tensor strategy);

    void updateRange(torch::Tensor strategy, size_t actionIdx);
    std::shared_ptr<DeepResolving> resolving = nullptr;

    torch::Tensor playerRange;
    torch::Tensor prevPlayerRange;
    torch::Tensor prevPlayerStrategy;
    torch::Tensor opponentCFV;
    int position = -1;
    int decisionCount = 0;
    int street = 1;
    int ourLastAction = -1;

    bool modelEvaluation = false;
};