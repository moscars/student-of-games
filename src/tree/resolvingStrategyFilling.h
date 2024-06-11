#pragma once

#include "../utils/publicBeliefState.h"
#include "../solving/deepResolving.h"
#include "../settings.h"

#include <memory>
#include <optional>

// Recursively walks the game tree and does resolving at every node
// This is then used to evaluate the exploitability of the generated strategy
class StrategyFilling {

public:
    void fillResolvingStrategy(std::shared_ptr<PublicBeliefState> root);

    int cnt = 0;

private:
    void fillDFS(std::shared_ptr<PublicBeliefState> node, 
                Player playerToFill, 
                torch::Tensor playerRange, 
                std::optional<torch::Tensor> opponentCFVs, 
                std::shared_ptr<DeepResolving> resolving,
                std::optional<int> ourLastAction,
                torch::Tensor opponentRangeEstimate);
    
    void fillPlayer(std::shared_ptr<PublicBeliefState> node, 
                    Player playerToFill, 
                    torch::Tensor playerRange, 
                    std::optional<torch::Tensor> opponentCFVs,
                    torch::Tensor opponentRangeEstimate);

    void fillOpponent(std::shared_ptr<PublicBeliefState> node, 
                    Player playerToFill, 
                    torch::Tensor playerRange, 
                    std::optional<torch::Tensor> opponentCFVs, 
                    std::shared_ptr<DeepResolving> resolving,
                    std::optional<int> ourLastAction,
                    torch::Tensor opponentRangeEstimate);

    void fillChance(std::shared_ptr<PublicBeliefState> node, 
                    Player playerToFill, 
                    torch::Tensor playerRange, 
                    std::optional<torch::Tensor> opponentCFVs, 
                    std::shared_ptr<DeepResolving> resolving,
                    std::optional<int> ourLastAction,
                    torch::Tensor opponentRangeEstimate);
};