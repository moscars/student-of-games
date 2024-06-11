#include "deepResolving.h"

void DeepResolving::resolve(std::shared_ptr<PublicBeliefState> tree,
                            torch::Tensor playerRange,
                            torch::Tensor opponentCounterFactualValues,
                            [[maybe_unused]] torch::Tensor opponentRangeEstimate
                ){
    assert(tree->getPlayer() != Player::CHANCE);

    playerStartingRange = playerRange.clone();

    TreeBuilder treeBuilder;
    subGameRoot = treeBuilder.buildTree(tree, settings::maxDepth + 2);
    // arbitrary inital strategy profile
    treeutils::fillUniformStrategy(subGameRoot, settings::maxDepth + 1);

    torch::Tensor uniform = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;
    torch::Tensor opponentRange = uniform.clone();//opponentRangeEstimate.clone() * 0.9 + uniform * 0.1;

    bool p1Playing = (tree->getPlayer() == Player::P1);
    int opponentIndex = static_cast<int>(utils::otherPlayer(tree->getPlayer()));

    for(iter = 1; iter <= settings::CFRIter; iter++){
        // get the values of the tree
        values(subGameRoot, p1Playing ? playerRange : opponentRange, p1Playing ? opponentRange : playerRange, 0);
        updateSubTreeStrategies(subGameRoot, 0);
        auto currentOpponentValue = subGameRoot->cfValues[opponentIndex].clone();
        opponentRange = rangeGadget(subGameRoot, currentOpponentValue, opponentCounterFactualValues);

        updateAverageCFV(subGameRoot, subGameRoot->getPlayer());
        trackPlayerBoardValue(subGameRoot, static_cast<int>(subGameRoot->getPlayer()));
        trackOpponentBoardValueWrapper(subGameRoot, opponentIndex);
    }
    opponentRangeEstimateSave = opponentRange.clone();

    normalizeSubTreeStrategies(subGameRoot, 0);
}

void DeepResolving::resolveStartOfGame(std::shared_ptr<PublicBeliefState> tree, torch::Tensor p1Range, torch::Tensor p2Range){
    assert(tree->getPlayer() != Player::CHANCE);

    TreeBuilder treeBuilder;
    subGameRoot = treeBuilder.buildTree(tree, settings::maxDepth + 2);
    // arbitrary inital strategy profile
    treeutils::fillUniformStrategy(subGameRoot, settings::maxDepth + 1);

    int opponentIndex = static_cast<int>(utils::otherPlayer(tree->getPlayer()));

    if(tree->getPlayer() == Player::P1){
        playerStartingRange = p1Range.clone();
    } else{
        playerStartingRange = p2Range.clone();
    }

    for(iter = 1; iter <= settings::CFRIter; iter++){
        // get the values of the tree
        values(subGameRoot, p1Range, p2Range, 0);

        updateSubTreeStrategies(subGameRoot, 0);

        updateAverageCFV(subGameRoot, subGameRoot->getPlayer());
        // updateAverageOpponentCFV(subGameRoot, opponentIndex);
        trackOpponentBoardValueWrapper(subGameRoot, opponentIndex);
        trackPlayerBoardValue(subGameRoot, static_cast<int>(subGameRoot->getPlayer()));
        updateAverageRootCFV(subGameRoot);
    }

    opponentRangeEstimateSave = (opponentIndex == 0) ? p2Range.clone() : p1Range.clone();

    normalizeSubTreeStrategies(subGameRoot, 0);
}


void DeepResolving::values(std::shared_ptr<PublicBeliefState> node,
                            torch::Tensor player1Range,
                            torch::Tensor player2Range,
                            int depth
                            ){

    if(node->isTerminal()){
        torch::Tensor values = torch::zeros({2, settings::numPossibleHands});
        if(node->getActionToThisNode() == ActionType::FOLD){
            auto valueP1 = netHolder::equityCache->getTerminalEquityFold(node->getBoard(), player2Range);
            auto valueP2 = netHolder::equityCache->getTerminalEquityFold(node->getBoard(), player1Range);
            values = torch::stack({valueP1, valueP2});
            Player currentPlayer = node->getPlayer();
            assert(currentPlayer != Player::CHANCE);
            int opponent_index = static_cast<int>(utils::otherPlayer(currentPlayer));
            values[opponent_index] *= -1;
        } else{
            values = netHolder::equityCache->getTerminalEquityCall(node->getBoard(), player1Range, player2Range);
        }

        values *= node->getPotSize();
        node->cfValues = values.clone();

        if(iter >= settings::warmupCFRIter){
            if(node->cumulativeCFValues.numel() == 0){
                node->cumulativeCFValues = values.clone();
            } else{
                node->cumulativeCFValues += values;
            }
        }

    } else if (depth >= settings::maxDepth && node->getPlayer() != Player::CHANCE){
        Inference inference;
        auto values = inference.evaluate(node, player1Range, player2Range, modelIndex);
        values *= node->getPotSize();
        node->cfValues = values.clone();

        if(iter >= settings::warmupCFRIter){
            if(node->cumulativeCFValues.numel() == 0){
                node->cumulativeCFValues = values.clone();
            } else{
                node->cumulativeCFValues += values;
            }
        }

        // DeepResolving dr;
        // dr.resolveStartOfGame(node, player1Range, player2Range);
        // node->cfValues = dr.getRootCFV();
    } else {
        torch::Tensor cfValues = torch::zeros({2, settings::numPossibleHands});
        for(int i = 0; i < static_cast<int>(node->getChildren().size()); i++){

            // update the range of the acting player
            Player actingPlayer = node->getPlayer();
            auto newP1Range = player1Range.clone();
            auto newP2Range = player2Range.clone();
            if(actingPlayer == Player::P1){
                newP1Range = player1Range * node->strategy.select(1, i);
            } else if (actingPlayer == Player::P2){
                newP2Range = player2Range * node->strategy.select(1, i);
            } else{
                newP1Range = player1Range * node->strategy.select(1, i);
                newP2Range = player2Range * node->strategy.select(1, i);
            }

            values(node->getChildren()[i], newP1Range, newP2Range, depth + 1);
            auto childValues = node->getChildren()[i]->cfValues;

            if(actingPlayer == Player::CHANCE){
                cfValues += childValues;
            } else{
                int actingPlayerIndex = static_cast<int>(actingPlayer);
                int otherPlayerIndex = static_cast<int>(utils::otherPlayer(actingPlayer));
                cfValues[actingPlayerIndex] += childValues[actingPlayerIndex] * node->strategy.select(1, i);
                cfValues[otherPlayerIndex] += childValues[otherPlayerIndex];
            }
            
        }
        node->cfValues = cfValues.clone();

        if(iter >= settings::warmupCFRIter){
            if(node->cumulativeCFValues.numel() == 0){
                node->cumulativeCFValues = cfValues.clone();
            } else{
                node->cumulativeCFValues += cfValues;
            }
        }
    }
}

torch::Tensor DeepResolving::getPlayerCFV(size_t actionIdx){
    assert(iter == settings::CFRIter + 1);

    auto value = subGameRoot->getChildren()[actionIdx]->cumulativeCFValues[static_cast<int>(subGameRoot->getPlayer())].clone();
    value /= (settings::CFRIter - settings::warmupCFRIter);

    return value;
}

void DeepResolving::updateSubTreeStrategies(std::shared_ptr<PublicBeliefState> node, int depth){
    if(node->isTerminal()) return;
    if(depth == settings::maxDepth) return;

    if(node->getPlayer() == Player::CHANCE){
        for(auto child : node->getChildren()){
            updateSubTreeStrategies(child, depth + 1);
        }
    } else {

        int actions_count = static_cast<int>(node->getChildren().size());
        int infosets_count = settings::numPossibleHands;

        int player = static_cast<int>(node->getPlayer());
        torch::Tensor current_regrets = torch::zeros({actions_count, infosets_count});
        for(size_t childIdx = 0; childIdx < node->getChildren().size(); childIdx++){
            std::shared_ptr<PublicBeliefState> child = node->getChildren()[childIdx];
            // the regret for each infoset given that I took action A is the 
            current_regrets[childIdx] = child->cfValues[player] - node->cfValues[player];
        }
        current_regrets = current_regrets.t();

        // Sum cumulative regrets
        if(node->regrets.numel() == 0){
            node->regrets = torch::full({infosets_count, actions_count}, settings::regretEpsilon);
        }

        node->regrets += current_regrets;
        node->regrets = torch::where(node->regrets > settings::regretEpsilon, node->regrets, settings::regretEpsilon);

        // update strategy with regret matching
        torch::Tensor positiveRegrets = node->regrets.clone();
        positiveRegrets = torch::where(positiveRegrets > settings::regretEpsilon, positiveRegrets, settings::regretEpsilon);

        torch::Tensor regretSum = positiveRegrets.sum(1); // CxA -> C -> The total regret for each infoset

        auto current_strategy = positiveRegrets.clone();
        // normalize the strategy (each row -> strategy for one infoset -> should sum to one)
        current_strategy /= regretSum.view({settings::numPossibleHands, 1});

        node->strategy = current_strategy.clone();

        if(iter >= settings::warmupCFRIter){
            if(node->cumulativeStrategy.numel() == 0){
                node->cumulativeStrategy = torch::zeros({infosets_count, actions_count});
            }
            node->cumulativeStrategy += node->strategy;
        }

        for(auto child : node->getChildren()){
            updateSubTreeStrategies(child, depth + 1);
        }
    }

}

torch::Tensor DeepResolving::rangeGadget(std::shared_ptr<PublicBeliefState> node, torch::Tensor currentOpponentValue, torch::Tensor initialOpponentCFV){
    if(gadgetPlayStrategy.numel() == 0){
        // initially only play terminate and see what regret is generated
        // which informs which hands to play in the subgame
        gadgetPlayStrategy = torch::ones({settings::numPossibleHands}) / 2;
        // The gadget game regrets also needs to be initialized
        cumulativePlayRegret = torch::zeros({settings::numPossibleHands});
        cumulativeTerminateRegret = torch::zeros({settings::numPossibleHands});
    } else{
        // do regret matching
        // we use cfr+ in reconstruction
        torch::Tensor positivePlayRegret = torch::where(cumulativePlayRegret > settings::regretEpsilon, cumulativePlayRegret, settings::regretEpsilon);
        torch::Tensor positiveTerminateRegret = torch::where(cumulativeTerminateRegret > settings::regretEpsilon, cumulativeTerminateRegret, settings::regretEpsilon);
        cumulativePlayRegret = positivePlayRegret.clone();
        cumulativeTerminateRegret = positiveTerminateRegret.clone();

        gadgetPlayStrategy = positivePlayRegret / (positivePlayRegret + positiveTerminateRegret);

        // filter out impossible hands given the board
        if(node->getBoard().size() > 0){
            for(int handIdx = 0; handIdx < settings::numPossibleHands; handIdx++){
                std::vector<Card> hand = settings::possibleHands[handIdx];
                if(utils::handBlockedByBoard(hand, node->getBoard())){
                    gadgetPlayStrategy[handIdx] = 0;
                }
            }
        }
    }

    if(previousOpponentValue.numel() == 0){
        previousOpponentValue = currentOpponentValue.clone();
    }

    torch::Tensor playRange = gadgetPlayStrategy.clone();
    torch::Tensor terminateRange = 1 - playRange;
    torch::Tensor gadgetValue = playRange * previousOpponentValue + terminateRange * initialOpponentCFV; // EV of the gadget game

    if(previousGadgetValue.numel() == 0){
        previousGadgetValue = gadgetValue.clone();
    }

    cumulativeTerminateRegret += (initialOpponentCFV - previousGadgetValue);
    cumulativePlayRegret += (currentOpponentValue - gadgetValue);

    previousGadgetValue = gadgetValue.clone();
    previousOpponentValue = currentOpponentValue.clone();

    return gadgetPlayStrategy;
}

void DeepResolving::normalizeSubTreeStrategies(std::shared_ptr<PublicBeliefState> node, int depth){
    if(node->isTerminal()) return;
    if(depth == settings::maxDepth) return;

    if(node->getPlayer() == Player::CHANCE){
        for(auto child : node->getChildren()){
            normalizeSubTreeStrategies(child, depth + 1);
        }
    } else{
        for(int i = 0; i < node->strategy.size(0); i++){
            node->strategy[i] = node->cumulativeStrategy[i] / (settings::CFRIter - settings::warmupCFRIter);
            node->strategy[i] /= torch::sum(node->strategy[i]);
        }

        for(auto child : node->getChildren()){
            normalizeSubTreeStrategies(child, depth + 1);
        }
    }

}

void DeepResolving::updateAverageCFV(std::shared_ptr<PublicBeliefState> node, Player resolvingPlayer){
    if(iter <= settings::warmupCFRIter) return;

    // needs to have for the resolving player when considering the first node
    updateAveragePlayerCFV(node, static_cast<int>(resolvingPlayer));
    updateAverageOpponentCFV(node, static_cast<int>(utils::otherPlayer(resolvingPlayer)));
}

void DeepResolving::updateAverageRootCFV(std::shared_ptr<PublicBeliefState> node){
    if(iter <= settings::warmupCFRIter) return;

    if(averageRootCFV.numel() == 0){
        averageRootCFV = torch::zeros({2, settings::numPossibleHands});
    }

    averageRootCFV += node->cfValues.clone();
}

void DeepResolving::updateAveragePlayerCFV(std::shared_ptr<PublicBeliefState> node, int playerIndex){
    if(iter <= settings::warmupCFRIter) return;

    if(averagePlayerCFV.numel() == 0){
        averagePlayerCFV = torch::zeros({settings::numPossibleHands});
    }

    averagePlayerCFV += node->cfValues[playerIndex].clone();
}

void DeepResolving::updateAverageOpponentCFV(std::shared_ptr<PublicBeliefState> node, int opponentIndex){
    if(iter <= settings::warmupCFRIter) return;

    if(averageOpponentCFV.numel() == 0){
        averageOpponentCFV = torch::zeros({static_cast<int>(node->getChildren().size()), settings::numPossibleHands});
    }

    for(size_t childIdx = 0; childIdx < node->getChildren().size(); childIdx++){
        averageOpponentCFV[childIdx] += node->getChildren()[childIdx]->cfValues[opponentIndex].clone();
    }
}

torch::Tensor DeepResolving::getPlayerCFV(){
    assert(iter == settings::CFRIter + 1);

    // TODO
    // divide the CFVs by the average strategy

    // divide the averagePlayerCFV with the average strategy
    return averagePlayerCFV / (settings::CFRIter - settings::warmupCFRIter);
}

torch::Tensor DeepResolving::getRootCFV(){
    assert(iter == settings::CFRIter + 1);
    return averageRootCFV / (settings::CFRIter - settings::warmupCFRIter);
}

torch::Tensor DeepResolving::getOpponentCFVBound(size_t actionIdx){
    assert(iter == settings::CFRIter + 1);

    // TODO
    // divide the CFVs by the average strategy

    // divide the averageOpponentCFV with the average strategy
    auto average = averageOpponentCFV.clone()[actionIdx];// / (settings::CFRIter - settings::warmupCFRIter);// (C)

    // averageOpponentCFV (C)
    //if(subGameRoot->getChildren()[actionIdx]->strategy.numel() > 0){
    //auto scaler = getStrategy().select(1, actionIdx); // (C, A)
    // auto scaler = subGameRoot->getChildren()[actionIdx]->ranges[static_cast<int>(utils::otherPlayer(subGameRoot->getPlayer()))].clone(); // (C)
    // average *= scaler;
    auto rootStrategy = getStrategy(); // (C, A)

    average /= (settings::CFRIter - settings::warmupCFRIter); // (C)
    
    auto playerRange = playerStartingRange * rootStrategy.select(1, actionIdx); // (C)
    assert(torch::sum(playerRange).item<float>() > 0);
    average /= torch::sum(playerRange); // (C)

    return average;
}

torch::Tensor DeepResolving::getStrategy(){
    return subGameRoot->strategy.clone();
}

torch::Tensor DeepResolving::getOpponentRangeEstimate(){
    //return subGameRoot->ranges[static_cast<int>(utils::otherPlayer(subGameRoot->getPlayer()))].clone();
    return opponentRangeEstimateSave;
}

void DeepResolving::setModelIndex(int index){
    modelIndex = index;
}

void DeepResolving::trackOpponentBoardValueWrapper(std::shared_ptr<PublicBeliefState> node, int opponentIndex){
    assert(node->getPlayer() == subGameRoot->getPlayer());
    // enumerate over all possible actions
    for(int actionIdx = 0; actionIdx < static_cast<int>(node->getChildren().size()); actionIdx++){
        trackOpponentBoardValue(node->getChildren()[actionIdx], opponentIndex, actionIdx);
    }
}


void DeepResolving::trackOpponentBoardValue(std::shared_ptr<PublicBeliefState> node, int opponentIndex, int ourLastAction){
    if(iter <= settings::warmupCFRIter) return;
    if(node->isTerminal()) return;

    if(node->getPlayer() == Player::CHANCE){
        if(!opponentBoardValue.count(ourLastAction)){
            opponentBoardValue[ourLastAction] = torch::zeros({settings::deckSize, settings::numPossibleHands});
        }
        opponentBoardValueCnt[ourLastAction]++;
        for(size_t actionIdx = 0; actionIdx < node->getChildren().size(); actionIdx++){
            opponentBoardValue[ourLastAction][actionIdx] += node->getChildren()[actionIdx]->cfValues[opponentIndex].clone();
        }
    } else if(static_cast<int>(node->getPlayer()) == opponentIndex){
        for(auto child : node->getChildren()){
            trackOpponentBoardValue(child, opponentIndex, ourLastAction);
        }
    } else{
        return;
    }
}

void DeepResolving::trackPlayerBoardValue(std::shared_ptr<PublicBeliefState> node, int playerIndex){
    if(iter <= settings::warmupCFRIter) return;
    if(node->isTerminal()) return;

    if(node == subGameRoot){
        for(auto child : node->getChildren()){
            trackPlayerBoardValue(child, playerIndex);
        }
    }

    if(node->getPlayer() == Player::CHANCE){
        if(playerBoardValue.numel() == 0){
            playerBoardValue = torch::zeros({settings::deckSize, settings::numPossibleHands});
        }

        if(playerBoardRootValue.numel() == 0){
            playerBoardRootValue = torch::zeros({settings::numPossibleHands});
        }

        playerBoardRootValue += node->cfValues[playerIndex].clone();

        for(size_t actionIdx = 0; actionIdx < node->getChildren().size(); actionIdx++){
            playerBoardValue[actionIdx] += node->getChildren()[actionIdx]->cfValues[playerIndex].clone();
        }
    } else if(static_cast<int>(node->getPlayer()) != playerIndex){
        for(auto child : node->getChildren()){
            trackPlayerBoardValue(child, playerIndex);
        }
    } else{
        return;
    }
}

torch::Tensor DeepResolving::getPlayerBoundOnBoard(size_t boardAction){
    assert(iter == settings::CFRIter + 1);
    assert(playerBoardValue.numel() > 0);

    return playerBoardValue[boardAction] / (settings::CFRIter - settings::warmupCFRIter);
}

torch::Tensor DeepResolving::getPlayerBoundOnBoardRoot(){
    assert(iter == settings::CFRIter + 1);
    assert(playerBoardRootValue.numel() > 0);

    return playerBoardRootValue / (settings::CFRIter - settings::warmupCFRIter);
}

torch::Tensor DeepResolving::getOpponentBoundOnBoard2(int ourLastAction, size_t boardAction, torch::Tensor chanceStrategy){
    assert(iter == settings::CFRIter + 1);
    assert(opponentBoardValue.count(ourLastAction));

    // keeping track of both our last action and the opponents last action is not necessary
    // but doing it guarantees that we update the CFVs the expected number of times
    // (seems to work anyways but keep it in since it makes more sense)
    assert(opponentBoardValueCnt[ourLastAction] == (settings::CFRIter - settings::warmupCFRIter));

    auto average = opponentBoardValue[ourLastAction].clone()[boardAction] / (settings::CFRIter - settings::warmupCFRIter);

    auto rootStrategy = getStrategy(); // (C, A)

    // assert(chanceStrategy[boardAction].item<float>() == 0);

    // chanceStrategy[boardAction] = 0;

    // torch::Tensor chanceStrategy1 = torch::ones({settings::deckSize}) / 4;
    // chanceStrategy1[boardAction] = 0;

    average /= torch::sum(playerStartingRange * rootStrategy.select(1, ourLastAction) * chanceStrategy);
    return average;
}