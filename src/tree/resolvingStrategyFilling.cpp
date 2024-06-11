#include "resolvingStrategyFilling.h"

void StrategyFilling::fillResolvingStrategy(std::shared_ptr<PublicBeliefState> root){
    auto uniformRange = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;

    std::shared_ptr<DeepResolving> resolving = std::make_shared<DeepResolving>();

    fillDFS(root, Player::P1, uniformRange, std::nullopt, resolving, std::nullopt, uniformRange);

    resolving = std::make_shared<DeepResolving>();
    resolving->resolveStartOfGame(root, uniformRange, uniformRange);

    torch::Tensor playerCFV = resolving->getPlayerCFV();

    for(size_t i = 0; i < root->getChildren().size(); i++){
        // The counterfactual value for the opponent (P1) is filled during resolving
        //torch::Tensor rangeEstimate = resolving->subGameRoot->getChildren()[i]->ranges[static_cast<int>(Player::P1)].clone();
        //auto rangeEstimate  
        fillDFS(root->getChildren()[i], Player::P2, uniformRange, playerCFV.clone(), resolving, std::nullopt, uniformRange);
    }
}

void StrategyFilling::fillDFS(std::shared_ptr<PublicBeliefState> node, 
                            Player playerToFill, 
                            torch::Tensor playerRange, 
                            std::optional<torch::Tensor> opponentCFVs, 
                            std::shared_ptr<DeepResolving> resolving,
                            std::optional<int> ourLastAction,
                            torch::Tensor opponentRangeEstimate){
    
    cnt++;
    if(cnt % 50 == 0) pr(cnt);
    
    if(node->isTerminal()) return;

    if(node->ranges.numel() == 0){
        node->ranges = torch::ones({2, settings::numPossibleHands}) / settings::numPossibleHands;
    }
    node->ranges[static_cast<int>(playerToFill)] = playerRange;

    if(node->cfValues.numel() == 0){
        node->cfValues = torch::zeros({2, settings::numPossibleHands});
    }

    if(opponentCFVs.has_value()) 
        node->cfValues[static_cast<int>(utils::otherPlayer(playerToFill))] = *opponentCFVs;

    // is it a chance node
    if(node->getPlayer() == Player::CHANCE){
        // handle chance player
        fillChance(node, playerToFill, playerRange, opponentCFVs, resolving, ourLastAction, opponentRangeEstimate);
    } else if(node->getPlayer() == playerToFill){
        // handle currentPlayer
        fillPlayer(node, playerToFill, playerRange, opponentCFVs, opponentRangeEstimate);
    } else{
        // handle opponent
        fillOpponent(node, playerToFill, playerRange, opponentCFVs, resolving, ourLastAction, opponentRangeEstimate);
    }
}

void StrategyFilling::fillPlayer(std::shared_ptr<PublicBeliefState> node, 
                                Player playerToFill, 
                                torch::Tensor playerRange, 
                                std::optional<torch::Tensor> opponentCFVs,
                                torch::Tensor opponentRangeEstimate){

    std::shared_ptr<DeepResolving> resolving = std::make_shared<DeepResolving>();

    //ppr("About to Resolve node with player:", playerToFill);

    assert(node->getPlayer() == playerToFill);

    auto uniformRange = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;

    if(!opponentCFVs.has_value()){
        assert(!node->hasParentNode()); // We must be dealing with the root node and player 1 must act first
        assert(node->getPlayer() == Player::P1);
        resolving->resolveStartOfGame(node, uniformRange, uniformRange);
    } else{
        resolving->resolve(node, playerRange, *opponentCFVs, opponentRangeEstimate);
    }

    node->strategy = resolving->getStrategy();

    std::vector<torch::Tensor> values;
    for(size_t i = 0; i < node->getChildren().size(); i++){
        values.push_back(resolving->getOpponentCFVBound(i));
    }

    torch::Tensor newOpponentEstimate = resolving->getOpponentRangeEstimate();

    for(size_t i = 0; i < node->getChildren().size(); i++){
        torch::Tensor rangeAfterAction = node->strategy.select(1, i) * playerRange;
        assert(torch::sum(rangeAfterAction).item<float>() > 0);
        rangeAfterAction /= torch::sum(rangeAfterAction);
        fillDFS(node->getChildren()[i], playerToFill, rangeAfterAction, values[i].clone(), resolving, i, newOpponentEstimate);
    }
}

void StrategyFilling::fillOpponent(std::shared_ptr<PublicBeliefState> node, 
                                    Player playerToFill, 
                                    torch::Tensor playerRange, 
                                    std::optional<torch::Tensor> opponentCFVs, 
                                    std::shared_ptr<DeepResolving> resolving,
                                    std::optional<int> ourLastAction,
                                    torch::Tensor opponentRangeEstimate){
    for(auto child : node->getChildren()){
        fillDFS(child, playerToFill, playerRange, opponentCFVs, resolving, ourLastAction, opponentRangeEstimate);
    }
}

void StrategyFilling::fillChance(std::shared_ptr<PublicBeliefState> node, 
                                Player playerToFill, 
                                torch::Tensor playerRange, 
                                std::optional<torch::Tensor> opponentCFVs, 
                                std::shared_ptr<DeepResolving> resolving,
                                std::optional<int> ourLastAction,
                                torch::Tensor opponentRangeEstimate){
    if(node->isTerminal()) return;
    assert(opponentCFVs.has_value());
    assert(node->getPlayer() == Player::CHANCE);
    assert(resolving != nullptr);
    assert(ourLastAction.has_value());

    std::vector<torch::Tensor> values;
    for(size_t i = 0; i < node->getChildren().size(); i++){
        // check the child node number of board cards
        values.push_back(resolving->getOpponentBoundOnBoard2(*ourLastAction, i, node->strategy.select(1, i)));
    }

    for(size_t i = 0; i < node->getChildren().size(); i++){
        torch::Tensor rangeAfterChanceCard = playerRange.clone();
        rangeAfterChanceCard *= node->strategy.select(1, i);
        // rangeAfterChanceCard *= node->strategy.select(1, i);
        // rangeAfterChanceCard[i] = 0;
        assert(torch::sum(rangeAfterChanceCard).item<float>() > 0);
        rangeAfterChanceCard /= torch::sum(rangeAfterChanceCard);
        fillDFS(node->getChildren()[i], playerToFill, rangeAfterChanceCard, values[i].clone(), nullptr, std::nullopt, opponentRangeEstimate);
    }
}
