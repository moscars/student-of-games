#include "continualResolving.h"

ContinualResolving::ContinualResolving(){
    playerRange = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;
}

torch::Tensor ContinualResolving::getPlayerRange() const {
    return playerRange.clone();
}

int ContinualResolving::getPosition() const {
    return position;
}

void ContinualResolving::setModelEvaluation(bool modelEvaluation_){
    modelEvaluation = modelEvaluation_;
}

GameState ContinualResolving::actUniformRandom(GameState state) {
    assert(resolving == nullptr);
    if(position == -1){
        position = static_cast<int>(state.actingPlayer);
    } else{
        assert(position == static_cast<int>(state.actingPlayer));
    }

    auto uniformRange = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;

    // zero out the board cards
    for(int i = 0; i < settings::numPossibleHands; i++){
        if(utils::handBlockedByBoard(settings::possibleHands[i], state.board)){
            uniformRange[i] = 0;
        }
    }

    uniformRange /= torch::sum(uniformRange);
    prevPlayerRange = uniformRange.clone();
    playerRange = uniformRange.clone();

    auto tmp = treeutils::copyNodeFromState(state);
    TreeBuilder builder;
    auto node = builder.buildTree(tmp, 3);
    // check how many children the node has
    size_t numChildren = node->getChildren().size();
    int actionIdx = torch::randint(0, numChildren, {1}).item<int>();
    prevPlayerStrategy = torch::ones({settings::numPossibleHands, static_cast<int>(numChildren)}) / static_cast<int>(numChildren);

    std::shared_ptr<PublicBeliefState> finalNode = node->getChildren()[actionIdx];

    GameState finalState = treeutils::copyStateFromNode(finalNode);
    finalState.actionIdxToThisState = actionIdx;
    return finalState;
}


torch::Tensor ContinualResolving::getOpponentCFVBoundBoard(const std::vector<Card> & board, bool adjustRange){
    int publicCardIdx = -1;
    int index = 0;
    for(int i = 0; i < settings::numPossibleHands; i++){
        if(board.size() == 2 && Card(settings::deckCards[i]) == board[0]){
            continue;
        }

        if(Card(settings::deckCards[i]) == board.back()){
            publicCardIdx = index;
            break;
        }
        index++;
    }

    // (hands, actions)
    auto chanceCardStrategy = torch::ones({settings::numPossibleHands, settings::deckSize});
    // if the board has one card
    if(board.size() >= 1){
        chanceCardStrategy /= (settings::deckSize - 2 * settings::handSize);
    } else{
        assert(false);
    }

    // zero out the probability of the public cards
    for(int i = 0; i < settings::numPossibleHands; i++){
        if(utils::handBlockedByBoard(settings::possibleHands[i], board)){
            chanceCardStrategy[i] = 0;
        }
    }

    if(adjustRange){
        playerRange *= chanceCardStrategy.select(1, publicCardIdx);
        // playerRange[publicCardIdx] = 0;
        playerRange /= torch::sum(playerRange);
    }

    return resolving->getOpponentBoundOnBoard2(ourLastAction, publicCardIdx, chanceCardStrategy.select(1, publicCardIdx));
}


GameState ContinualResolving::act(GameState state){
    if(position == -1){
        position = static_cast<int>(state.actingPlayer);
    } else{
        assert(resolving != nullptr);
        assert(position == static_cast<int>(state.actingPlayer));
    }

    // check if the street has changed
    if(street != state.street){
        assert(state.street == street + 1);
        street = state.street;

        opponentCFV = getOpponentCFVBoundBoard(state.board, true);
    }

    // resolve the tree
    auto uniformRange = torch::ones({settings::numPossibleHands}) / settings::numPossibleHands;

    if(decisionCount == 0){
        resolving = std::make_shared<DeepResolving>();
        if(modelEvaluation) resolving->setModelIndex(position);
        if(state.isStartOfGame){
            assert(state.actingPlayer == Player::P1);
            assert(state.street == 1);
            assert(state.board.empty());
            // copy over the important information
            std::shared_ptr<PublicBeliefState> tree = treeutils::copyNodeFromState(state);
            resolving->resolveStartOfGame(tree, uniformRange, uniformRange);
        } else{
            assert(state.actingPlayer == Player::P2);
            assert(position == 1);
            DeepResolving p1Resolving;
            if(modelEvaluation) p1Resolving.setModelIndex(position);

            std::shared_ptr<PublicBeliefState> root = std::make_shared<PublicBeliefState>();
            root->setPlayer(Player::P1);
            root->setBets({settings::ante, settings::ante});
            root->setActionToThisNode(ActionType::START);
            root->setTerminal(false);
            root->setDepth(0);
            root->setStreet(1);

            bool temp = settings::selfPlay;
            if(temp) settings::selfPlay = false;
            p1Resolving.resolveStartOfGame(root, uniformRange, uniformRange);
            if(temp) settings::selfPlay = true;
            
            opponentCFV = p1Resolving.getPlayerCFV();

            std::shared_ptr<PublicBeliefState> tree = treeutils::copyNodeFromState(state); 

            resolving->resolve(tree, uniformRange, opponentCFV, uniformRange);
        }

    } else{
        resolving = std::make_shared<DeepResolving>();
        if(modelEvaluation) resolving->setModelIndex(position);
        std::shared_ptr<PublicBeliefState> tree = treeutils::copyNodeFromState(state); 
        resolving->resolve(tree, playerRange, opponentCFV, uniformRange);
    }

    // average the strategies (This is already done in getStrategy)
    // sample an action (a)
    size_t actionIdx = sampleAction(resolving->getStrategy());
    ourLastAction = static_cast<int>(actionIdx);
    decisionCount++;
    // update the range based on the chosen action (a)
    // normalize the range
    // what was our value before making the action

    updateRange(resolving->getStrategy(), actionIdx);
    // Average the counterfactual values after action (a)
    opponentCFV = resolving->getOpponentCFVBound(actionIdx);

    // tree node that we have reached
    std::shared_ptr<PublicBeliefState> finalNode = resolving->subGameRoot->getChildren()[actionIdx];

    GameState finalState = treeutils::copyStateFromNode(finalNode);
    finalState.actionIdxToThisState = actionIdx;
    return finalState;
}

float ContinualResolving::findAIVATValue(size_t actionIdx, torch::Tensor p1Range, torch::Tensor p2Range, int modelToUse){
    assert(resolving != nullptr);
    std::vector<torch::Tensor> equities;

    // to through the resolving children
    for(size_t i = 0; i < resolving->subGameRoot->getChildren().size(); i++){
        auto child = resolving->subGameRoot->getChildren()[i];

        torch::Tensor values = torch::zeros({2, settings::numPossibleHands});

        if(child->isTerminal()){
            if(child->getActionToThisNode() == ActionType::FOLD){
                auto valueP1 = netHolder::equityCache->getTerminalEquityFold(child->getBoard(), p2Range);
                auto valueP2 = netHolder::equityCache->getTerminalEquityFold(child->getBoard(), p1Range);
                values = torch::stack({valueP1, valueP2});
                Player currentPlayer = child->getPlayer();
                assert(currentPlayer != Player::CHANCE);
                int opponent_index = static_cast<int>(utils::otherPlayer(currentPlayer));
                values[opponent_index] *= -1;
            } else{
                values = netHolder::equityCache->getTerminalEquityCall(child->getBoard(), p1Range, p2Range);
            }
        } else{
            Inference inference;
            torch::Tensor equity = inference.evaluate(child, p1Range, p2Range, modelToUse);
            values = equity;
        }

        values *= child->getPotSize();
        equities.push_back(values[position]);
    }

    auto ourRange = (position == 0) ? p1Range : p2Range;

    // get the strategy
    torch::Tensor strategy = resolving->getStrategy();
    auto weightedStrategy = strategy * ourRange.view({settings::numPossibleHands, 1});

    // get the probability of playing each action
    std::vector<float> actionProbabilities;
    for(size_t i = 0; i < equities.size(); i++){
        auto strategyHere = weightedStrategy.select(1, i);
        actionProbabilities.push_back((torch::sum(strategyHere)/ torch::sum(weightedStrategy)).item<float>());
    }

    float probSum = std::accumulate(actionProbabilities.begin(), actionProbabilities.end(), 0.0);
    assert(std::abs(probSum - 1) < 1e-5);

    std::vector<float> values;
    for(size_t i = 0; i < equities.size(); i++){
        auto rangeHere = ourRange * strategy.select(1, i);
        assert(torch::sum(rangeHere).item<float>() > 0);
        rangeHere /= torch::sum(rangeHere);
        values.push_back(torch::dot(equities[i], rangeHere).item<float>());
    }

    // expectedValue
    float expectedValue = 0;
    for(size_t i = 0; i < equities.size(); i++){
        expectedValue += actionProbabilities[i] * values[i];
    }

    // actual value
    auto rangeHere = ourRange * strategy.select(1, actionIdx);
    assert(torch::sum(rangeHere).item<float>() > 0);
    rangeHere /= torch::sum(rangeHere);
    assert(actionIdx < equities.size());
    float actualValue = torch::dot(equities[actionIdx], rangeHere).item<float>();

    return expectedValue - actualValue;
}

float ContinualResolving::getAIVatFromBoardAction(size_t actionIdx, const std::vector<Card> & board){
    auto valueBefore = resolving->getPlayerBoundOnBoardRoot();
    auto valueAfter = resolving->getPlayerBoundOnBoard(actionIdx);

    // (hands, actions)
    auto chanceCardStrategy = torch::ones({settings::numPossibleHands, settings::deckSize});
    // if the board has one card
    chanceCardStrategy /= (settings::deckSize - 2 * settings::handSize);

    // zero out the probability of the public cards

    for(int i = 0; i < settings::numPossibleHands; i++){
        if(utils::handBlockedByBoard(settings::possibleHands[i], board)){
            chanceCardStrategy[i] = 0;
        }
    }

    auto rangeAfter = playerRange * chanceCardStrategy.select(1, actionIdx);

    std::cout << "Value before: " << valueBefore << std::endl;
    std::cout << "Range before: " << playerRange << std::endl;

    // print the board
    std::cout << "Board: ";
    for(const Card c : board){
        std::cout << c << " ";
    }
    std::cout << std::endl;


    std::cout << "Value after: " << valueAfter << std::endl;
    std::cout << "Range after: " << rangeAfter << std::endl;

    float vb = torch::dot(valueBefore, rangeAfter).item<float>();
    float va = torch::dot(valueAfter, rangeAfter).item<float>();

    return vb - va;
}


size_t ContinualResolving::sampleAction(torch::Tensor strategy){
    // Strategy has dimensions Cards x Actions
    // The probability of taking each action in the strategy is given by the sum of each column
    // divided by the sum of the entire strategy
    // check that the sum of each strategy for each card sums to one

    // pr("Sampling action from strategy");
    // pr(strategy);

    int numActions = strategy.size(1);
    torch::Tensor uniform = torch::ones({settings::numPossibleHands, numActions}) / numActions;
    
    torch::Tensor selfPlayStrategy;
    if(settings::selfPlay){
        selfPlayStrategy = (1 - settings::selfPlayEpsilon) * strategy + settings::selfPlayEpsilon * uniform;
    } else{
        selfPlayStrategy = strategy;
    }

    assert(torch::allclose(torch::sum(selfPlayStrategy, 1), torch::ones({settings::numPossibleHands})));

    selfPlayStrategy *= playerRange.view({settings::numPossibleHands, 1});
    torch::Tensor actionProbabilities = torch::sum(selfPlayStrategy, 0) / torch::sum(selfPlayStrategy);

    assert(std::abs(torch::sum(actionProbabilities).item<float>() - 1) < 1e-5);

    torch::Tensor cumulativeProbability = torch::cumsum(actionProbabilities, 0);

    float randomFloat = torch::rand({1}).item<float>();

    size_t actionIdx = 0;
    for(int i = 0; i < cumulativeProbability.size(0); i++){
        if(randomFloat < cumulativeProbability[i].item<float>()){
            actionIdx = i;
            break;
        }
    }

    return actionIdx;
}

void ContinualResolving::updateRange(torch::Tensor strategy, size_t actionIdx){
    // update playerRange given that we played the action in column actionIdx (strategy is hands x actions)
    // also normalizes the range

    torch::Tensor currentStrategy;
    if(settings::selfPlay){
        auto uniform = torch::ones({settings::numPossibleHands, strategy.size(1)}) / strategy.size(1);
        currentStrategy = (1 - settings::selfPlayEpsilon) * strategy + settings::selfPlayEpsilon * uniform;
    } else{
        currentStrategy = strategy;
    }

    prevPlayerRange = playerRange.clone();
    prevPlayerStrategy = currentStrategy.clone();
    
    // do bayes rule
    torch::Tensor newRange = playerRange * currentStrategy.select(1, actionIdx);
    // given bayes rule this should be divided by the probability of playing action actionIdx
    // but that just acts as a normalization constant and we want a normalized range
    // so instead just normalize the range
    assert(torch::sum(newRange).item<float>() > 0);
    playerRange = newRange / torch::sum(newRange);
}


torch::Tensor ContinualResolving::getPrevPlayerRange() const {
    return prevPlayerRange.clone();
}

torch::Tensor ContinualResolving::getPrevPlayerStrategy() const {
    return prevPlayerStrategy.clone();
}
