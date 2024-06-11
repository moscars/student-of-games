#include "dealer.h"

Dealer::Dealer(bool trackResults_, Player playerToTrack_){
    trackResults = trackResults_;
    playerToTrack = playerToTrack_;

    // Initialize the deck
    for(const std::string & c : settings::deckCards){
        deck.push_back(Card(c));
    }
}

Dealer::Dealer(){
    trackResults = false;
    playerToTrack = Player::CHANCE;

    // Initialize the deck
    for(const std::string & c : settings::deckCards){
        deck.push_back(Card(c));
    }
}


void Dealer::playHand(){
    bool uniformRandomOpponent = false;

    // make sure that the deck is full
    if(static_cast<int>(deck.size()) != settings::deckSize){
        deck.clear();
        for(const std::string & c : settings::deckCards){
            deck.push_back(Card(c));
        }
    }

    ContinualResolving player1;
    ContinualResolving player2;
    if(trackResults){
        player1.setModelEvaluation(true);
        player2.setModelEvaluation(true);
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(deck.begin(), deck.end(), g);

    GameState state;
    state.isStartOfGame = true;
    state.street = 1;
    state.actingPlayer = Player::P1;
    state.bets = {settings::ante, settings::ante};
    state.board.clear();
    state.isTerminal = false;
    state.actionToThisState = ActionType::START;

    Player lastActingPlayer = Player::CHANCE;
    
    float aivatTracker = 0;
    // torch::Tensor aivatTracker = torch::zeros({settings::numPossibleHands});

    int modelToUse = -1;
    if(playerToTrack == Player::P1){
        modelToUse = 0;
    } else if(playerToTrack == Player::P2){
        modelToUse = 1;
    }

    while(!state.isTerminal){
        lastActingPlayer = state.actingPlayer;

        if(state.actingPlayer == Player::P1){

            if(uniformRandomOpponent && playerToTrack != Player::P1){
                state = player1.actUniformRandom(state);
            } else{
                auto oldRange = player1.getPlayerRange();
                state = player1.act(state);

                if(trackResults && playerToTrack == Player::P1){
                    aivatTracker += player1.findAIVATValue(state.actionIdxToThisState, oldRange, player2.getPlayerRange(), modelToUse);
                }
            }

            settings::cnt++;
            std::cout << settings::cnt << std::endl;

        } else if(state.actingPlayer == Player::P2){

            if(uniformRandomOpponent && playerToTrack != Player::P2){
                state = player2.actUniformRandom(state);
            } else{
                auto oldRange = player2.getPlayerRange();
                state = player2.act(state);

                if(trackResults && playerToTrack == Player::P2){
                    aivatTracker += player2.findAIVATValue(state.actionIdxToThisState, player1.getPlayerRange(), oldRange, modelToUse);
                }
            }

            settings::cnt++;
            std::cout << settings::cnt << std::endl;

        } else{
            assert(state.actingPlayer == Player::CHANCE);
            state.actingPlayer = Player::P1;
            state.street++;
            state.actionToThisState = ActionType::CHANCE;


            // go through all possible cards that can be revealed

            // which are the cards in the deck

            if(trackResults){
                std::vector<Card> possibleCards;
                for(const Card & c : deck){
                    possibleCards.push_back(c);
                }

                // // go through each scenario
                auto rangeP1 = player1.getPlayerRange();
                auto rangeP2 = player2.getPlayerRange();
                std::vector<torch::Tensor> equities;
                for(const Card & c : possibleCards){
                    std::vector<Card> newBoard = state.board;
                    newBoard.push_back(c);

                    std::shared_ptr<PublicBeliefState> node = std::make_shared<PublicBeliefState>();
                    node->setPlayer(Player::P1);
                    node->setBets(state.bets);
                    node->setBoard(newBoard);
                    node->setActionToThisNode(ActionType::CHANCE);
                    node->setTerminal(false);

                    Inference Inference;
                    auto equity = Inference.evaluate(node, rangeP1, rangeP2, modelToUse);
                    float potSize = std::min(state.bets[0], state.bets[1]);
                    equity *= potSize;

                    equities.push_back(equity);
                }

                // get average of the equities
                torch::Tensor averageEquity = torch::zeros({2, settings::numPossibleHands});
                for(int i = 0; i < static_cast<int>(equities.size()); i++){
                    averageEquity += equities[i];
                }
                averageEquity /= static_cast<float>(equities.size());

                auto actualEquity = equities.back();
                
                if(playerToTrack == Player::P1){
                    aivatTracker += torch::dot(rangeP1, (averageEquity[0] - actualEquity[0])).item<float>();

                } else if(playerToTrack == Player::P2){
                    aivatTracker += torch::dot(rangeP2, (averageEquity[1] - actualEquity[1])).item<float>();
                }
            }

            state.board.push_back(deck.back());
            deck.pop_back();            
        }
    }

    if(trackResults){

        assert(playerToTrack != Player::CHANCE);
        float money = static_cast<float>(std::min(state.bets[0], state.bets[1]));
        // check if a player folded
        if(state.actionToThisState == ActionType::FOLD){
            assert(lastActingPlayer != Player::CHANCE);
            assert(std::abs(money - static_cast<float>(state.bets[static_cast<int>(lastActingPlayer)])) < 1e-4);
            // last acting player folded
            // see how much money changed hands

            // if this is the final round of betting
            if(state.board.size() == 2){
                // how likely were we to call?
                if(lastActingPlayer == Player::P1){
                    auto prevRange = player1.getPrevPlayerRange();
                    auto prevStrategy = player1.getPrevPlayerStrategy();

                    auto rangeP1 = player1.getPlayerRange();
                    auto rangeP2 = player2.getPlayerRange();

                    // first action is fold
                    auto foldStrategy = prevStrategy.select(1, 0);
                    float foldProb = (torch::sum(foldStrategy) / torch::sum(prevStrategy)).item<float>();
                    assert(torch::allclose(foldStrategy * prevRange / torch::sum(foldStrategy * prevRange), rangeP1));

                    // call is the second action
                    auto callStrategy = prevStrategy.select(1, 1);
                    float callProb = (torch::sum(callStrategy) / torch::sum(prevStrategy)).item<float>();

                    // call range
                    auto callRange = callStrategy * prevRange;
                    assert(torch::sum(callRange).item<float>() > 0);
                    callRange /= torch::sum(callRange);

                    // get the call EV
                    auto equity = netHolder::equityCache->getTerminalEquityCall(state.board, callRange, rangeP2);

                    float p1EV = torch::sum(equity[0] * callRange).item<float>();
                    // what is the value of calling
                    float callEV = p1EV * money;
                    float foldEV = -money;

                    float foldFraction = foldProb / (foldProb + callProb);
                    float callFraction = callProb / (foldProb + callProb);

                    float value = foldFraction * foldEV + callFraction * callEV;

                    if(playerToTrack == Player::P1){
                        IOResults.push_back(-money);
                        aivatResults.push_back(-money + aivatTracker);
                        FoldAndCallResults.push_back(value);
                        FoldAndRaiseResultsWithAIVAT.push_back(value + aivatTracker);
                    } else{
                        IOResults.push_back(money);
                        aivatResults.push_back(money + aivatTracker);
                        FoldAndCallResults.push_back(-value);
                        FoldAndRaiseResultsWithAIVAT.push_back(-value + aivatTracker);
                    }
                } else{
                    auto prevRange = player2.getPrevPlayerRange();
                    auto prevStrategy = player2.getPrevPlayerStrategy();

                    auto rangeP1 = player1.getPlayerRange();
                    auto rangeP2 = player2.getPlayerRange();

                    // first action is fold
                    auto foldStrategy = prevStrategy.select(1, 0);
                    float foldProb = (torch::sum(foldStrategy) / torch::sum(prevStrategy)).item<float>();
                    assert(torch::allclose(foldStrategy * prevRange / torch::sum(foldStrategy * prevRange), rangeP2));

                    // call is the second action
                    auto callStrategy = prevStrategy.select(1, 1);
                    float callProb = (torch::sum(callStrategy) / torch::sum(prevStrategy)).item<float>();

                    // call range
                    auto callRange = callStrategy * prevRange;
                    assert(torch::sum(callRange).item<float>() > 0);
                    callRange /= torch::sum(callRange);

                    // get the call EV
                    auto equity = netHolder::equityCache->getTerminalEquityCall(state.board, rangeP1, callRange);

                    float p2EV = torch::sum(equity[1] * callRange).item<float>();
                    // what is the value of calling
                    float callEV = p2EV * money;
                    float foldEV = -money;

                    float foldFraction = foldProb / (foldProb + callProb);
                    float callFraction = callProb / (foldProb + callProb);

                    float value = foldFraction * foldEV + callFraction * callEV;

                    if(playerToTrack == Player::P2){
                        IOResults.push_back(-money);
                        aivatResults.push_back(-money + aivatTracker);
                        FoldAndCallResults.push_back(value);
                        FoldAndRaiseResultsWithAIVAT.push_back(value + aivatTracker);
                    } else{
                        IOResults.push_back(money);
                        aivatResults.push_back(money + aivatTracker);
                        FoldAndCallResults.push_back(-value);
                        FoldAndRaiseResultsWithAIVAT.push_back(-value + aivatTracker);
                    }
                }
            } else{
                if(playerToTrack == lastActingPlayer){
                    IOResults.push_back(-money);
                    aivatResults.push_back(-money + aivatTracker);
                    FoldAndCallResults.push_back(-money);
                    FoldAndRaiseResultsWithAIVAT.push_back(-money + aivatTracker);
                } else{
                    IOResults.push_back(money);
                    aivatResults.push_back(money + aivatTracker);
                    FoldAndCallResults.push_back(money);
                    FoldAndRaiseResultsWithAIVAT.push_back(money + aivatTracker);
                }
            }


        } else{
            assert(state.actionToThisState == ActionType::CALL || state.actionToThisState == ActionType::TCHECK);
            auto rangeP1 = player1.getPlayerRange();
            auto rangeP2 = player2.getPlayerRange();

            assert(torch::sum(rangeP1).item<float>() > 0.999 && torch::sum(rangeP1).item<float>() < 1.001);
            assert(torch::sum(rangeP2).item<float>() > 0.999 && torch::sum(rangeP2).item<float>() < 1.001);

            auto equity = netHolder::equityCache->getTerminalEquityCall(state.board, rangeP1, rangeP2);

            float p1EV = torch::sum(equity[0] * rangeP1).item<float>();
            float p2EV = torch::sum(equity[1] * rangeP2).item<float>();

            assert(std::abs(p1EV + p2EV) < 1e-4); // should sum to zero

            // if last
            if(playerToTrack == Player::P1){
                IOResults.push_back(p1EV * money);
                aivatResults.push_back(p1EV * money + aivatTracker);
                FoldAndCallResults.push_back(p1EV * money);
                FoldAndRaiseResultsWithAIVAT.push_back(p1EV * money + aivatTracker);
            } else{
                IOResults.push_back(p2EV * money);
                aivatResults.push_back(p2EV * money + aivatTracker);
                FoldAndCallResults.push_back(p2EV * money);
                FoldAndRaiseResultsWithAIVAT.push_back(p2EV * money + aivatTracker);
            }
        }

    }

}

GameResults Dealer::getResults() const {
    return {IOResults, aivatResults, FoldAndCallResults, FoldAndRaiseResultsWithAIVAT};
}
