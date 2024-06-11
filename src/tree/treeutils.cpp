#include "treeutils.h"

namespace treeutils{
    void fillUniformStrategy(std::shared_ptr<PublicBeliefState> node, int maxDepth){
        if(node->isTerminal()) return;
        if(maxDepth == 0) return;
        // Fill the strategy with a uniform distribution
        std::vector<std::shared_ptr<PublicBeliefState>> children = node->getChildren();
        int numChildren = static_cast<int>(children.size());
        torch::Tensor strategy = torch::ones({settings::numPossibleHands, numChildren});
        if(node->getPlayer() == Player::CHANCE){
            // block off all cards that are on the board
            for(int i = 0; i < numChildren; i++){
                // what board card is this child representing
                for(const Card & boardCard : children[i]->getBoard()){
                    for(int handIdx = 0; handIdx < settings::numPossibleHands; handIdx++){
                        std::vector<Card> hand = settings::possibleHands[handIdx];
                        if(utils::handBlockedByBoard(hand, {boardCard})){
                            strategy[handIdx][i] = 0;
                        }
                    }
                }
            }
            // And given that each player holds one card each
            // the probability of each card appearing (in retrospect) is 1 / (deckSize - 2)
            strategy /= (settings::deckSize - 2 * settings::handSize);
        } else{
            strategy /= numChildren;
        }

        node->strategy = strategy;
        for(int i = 0; i < numChildren; i++){
            fillUniformStrategy(children[i], maxDepth - 1);
        }
    }


    std::shared_ptr<PublicBeliefState> copyNodeFromState(GameState state){
        std::shared_ptr<PublicBeliefState> node = std::make_shared<PublicBeliefState>();
        node->setPlayer(state.actingPlayer);
        node->setStreet(state.street);
        node->setBets(state.bets);
        node->setBoard(state.board);
        node->setTerminal(state.isTerminal);
        node->setActionToThisNode(state.actionToThisState);
        return node;
    }

    GameState copyStateFromNode(std::shared_ptr<PublicBeliefState> node){
        GameState state;
        state.actingPlayer = node->getPlayer();
        state.street = node->getStreet();
        state.bets = node->getBets();
        state.board = node->getBoard();
        state.isTerminal = node->isTerminal();
        state.isStartOfGame = (node->hasParentNode() == false);
        state.actionToThisState = node->getActionToThisNode();
        return state;
    }

};