#pragma once

#include "../utils/publicBeliefState.h"
#include "../tree/treeutils.h"
#include "../tree/treeBuilder.h"
#include "../game/equity.h"
#include "../nn/inference.h"

#include <memory>
#include <cassert>

#include <torch/torch.h>

class DeepResolving {

public:
    void resolve(std::shared_ptr<PublicBeliefState> tree, torch::Tensor playerRange, torch::Tensor opponentCounterFactualValues,
    [[maybe_unused]] torch::Tensor opponentRangeEstimate);
    void resolveStartOfGame(std::shared_ptr<PublicBeliefState> tree, torch::Tensor p1Range, torch::Tensor p2Range);
    void values(std::shared_ptr<PublicBeliefState> node, torch::Tensor player1Range, torch::Tensor player2Range, int depth);
    void updateSubTreeStrategies(std::shared_ptr<PublicBeliefState> node, int depth);
    torch::Tensor rangeGadget(std::shared_ptr<PublicBeliefState> node, torch::Tensor currentOpponentValue, torch::Tensor initialOpponentCFV);
    void normalizeSubTreeStrategies(std::shared_ptr<PublicBeliefState> node, int depth);

    std::shared_ptr<PublicBeliefState> subGameRoot;

    torch::Tensor getPlayerCFV();
    torch::Tensor getOpponentCFVBound(size_t actionIdx);
    torch::Tensor getStrategy();
    torch::Tensor getOpponentRangeEstimate();
    torch::Tensor getRootCFV();
    torch::Tensor getOpponentBoundOnBoard2(int ourLastAction, size_t boardAction, torch::Tensor chanceStrategy);
    torch::Tensor getPlayerCFV(size_t actionIdx);
    
    torch::Tensor getPlayerBoundOnBoard(size_t boardAction);
    torch::Tensor getPlayerBoundOnBoardRoot();

    void setModelIndex(int index);

private:
    torch::Tensor cumulativePlayRegret;
    torch::Tensor cumulativeTerminateRegret;
    torch::Tensor gadgetPlayStrategy;
    torch::Tensor gadgetGameValues;

    torch::Tensor opponentRangeEstimateSave;

    torch::Tensor averageOpponentCFV;
    torch::Tensor averagePlayerCFV;
    torch::Tensor averageRootCFV;
    torch::Tensor lastOpponentRange;

    torch::Tensor previousGadgetValue;
    torch::Tensor previousOpponentValue;

    torch::Tensor playerStartingRange;

    torch::Tensor playerBoardValue;
    torch::Tensor playerBoardRootValue;

    std::unordered_map<int, torch::Tensor> opponentBoardValue;
    std::unordered_map<int, int> opponentBoardValueCnt;

    std::optional<int> modelIndex;

    int iter;

    void updateAverageCFV(std::shared_ptr<PublicBeliefState> node, Player resolvingPlayer);
    void updateAverageOpponentCFV(std::shared_ptr<PublicBeliefState> node, int opponentIndex);
    void updateAveragePlayerCFV(std::shared_ptr<PublicBeliefState> node, int playerIndex);
    void updateAverageRootCFV(std::shared_ptr<PublicBeliefState> node);
    void trackOpponentBoardValue(std::shared_ptr<PublicBeliefState> node, int opponentIndex, int ourLastAction);
    void trackOpponentBoardValueWrapper(std::shared_ptr<PublicBeliefState> node, int opponentIndex);
    void trackPlayerBoardValue(std::shared_ptr<PublicBeliefState> node, int playerIndex);

};