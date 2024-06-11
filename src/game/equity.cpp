#include "equity.h"

// idea: count the number of times that a hand is better than another hand, given a board
// and then this can be cached

namespace equity{
    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCall(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2){
        
        if(board.size() == 0) {
            return getTerminalEquityCallNoBoard(board, rangeP1, rangeP2);
        }
        if(board.size() != static_cast<size_t>((settings::numStreets - 1))){
            if(board.size() == 1){
                return getTerminalEquityCallOneBoard(board, rangeP1, rangeP2);
            } else{
                assert(false);
            }
        }

        torch::Tensor equity = torch::zeros({2, settings::numPossibleHands});
        torch::Tensor count = torch::zeros({settings::numPossibleHands, settings::numPossibleHands});

        assert(rangeP1.size(0) == settings::numPossibleHands);
        assert(rangeP2.size(0) == settings::numPossibleHands);

        // assert(settings::deckSize == settings::numPossibleHands);
        assert(board.size() == static_cast<size_t>((settings::numStreets - 1)));

        for(int p1 = 0; p1 < settings::numPossibleHands; p1++){
            for(int p2 = 0; p2 < settings::numPossibleHands; p2++){
                if(p1 == p2) continue;
                std::vector<Card> p1Hand = settings::possibleHands[p1];
                std::vector<Card> p2Hand = settings::possibleHands[p2];
                // the hands cannot be blocking
                if(utils::handBlockedByBoard(p1Hand, p2Hand)) continue;
                // are the hands blocked by the board
                if(utils::handBlockedByBoard(p1Hand, board)) continue;
                if(utils::handBlockedByBoard(p2Hand, board)) continue;

                int p1Better = handComp::compareHands(board, p1Hand, p2Hand);

                if(p1Better == 1){
                    equity[0][p1] += rangeP2[p2];
                    equity[1][p2] -= rangeP1[p1];

                    count[p1][p2] += 1;

                } else if(p1Better == -1){
                    equity[0][p1] -= rangeP2[p2];
                    equity[1][p2] += rangeP1[p1];

                    count[p2][p1] += 1;
                }
            }
        }

        return {equity, count};
    }

    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCallNoBoard(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2){
        assert(board.size() == 0);
        torch::Tensor equity = torch::zeros({2, settings::numPossibleHands});
        torch::Tensor count = torch::zeros({settings::numPossibleHands, settings::numPossibleHands});
        // loop over all possible boards
        for(const std::string & boardCard : settings::deckCards){
            std::vector<Card> board_ = {Card(boardCard)};
            auto [equityBoard, cnt] = getTerminalEquityCallOneBoard(board_, rangeP1, rangeP2);
            // multiply by the probability of the board occuring
            // since each of the 2 players has a hand each, the probability of the board
            // is  1 / (possible cards - 2)
            equity += equityBoard * (1 / static_cast<float>(settings::deckSize - 2 * settings::handSize));
            count += cnt;
        }

        return {equity, count};
    }

    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityCallOneBoard(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2){
        assert(board.size() == 1);
        if(board.size() == static_cast<size_t>((settings::numStreets - 1))){
            return getTerminalEquityCall(board, rangeP1, rangeP2);
        }
        torch::Tensor equity = torch::zeros({2, settings::numPossibleHands});
        torch::Tensor count = torch::zeros({settings::numPossibleHands, settings::numPossibleHands});
        // loop over all possible boards
        for(const std::string & boardCard : settings::deckCards){
            if(Card(boardCard) == board[0]) continue;

            std::vector<Card> board_ = {board[0], Card(boardCard)};
            auto [equityBoard, cnt] = getTerminalEquityCall(board_, rangeP1, rangeP2);
            // multiply by the probability of the board occuring
            // since each of the 2 players has a hand each
            // and there is one card already on the board, the probability of the board
            // is  1 / (possible cards - 3)
            equity += equityBoard * (1 / static_cast<float>(settings::deckSize - 2 * settings::handSize));

            count += cnt;
        }

        return {equity, count};
    }

    std::pair<torch::Tensor, torch::Tensor> getTerminalEquityFold(const std::vector<Card> & board, torch::Tensor rangeOpponent){
        torch::Tensor equity = torch::zeros({settings::numPossibleHands});
        torch::Tensor count = torch::zeros({settings::numPossibleHands, settings::numPossibleHands});

        for(int handIdx = 0; handIdx < settings::numPossibleHands; handIdx++){
            std::vector<Card> hand = settings::possibleHands[handIdx];
            // if p1 is not blocked by the board
            if(utils::handBlockedByBoard(hand, board)) continue;

            float counterfactualValue = 0;
            for(int opponentHandIdx = 0; opponentHandIdx < settings::numPossibleHands; opponentHandIdx++){
                std::vector<Card> opponentHand = settings::possibleHands[opponentHandIdx];
                // if I block your hand
                if(utils::handBlockedByBoard(opponentHand, hand)) continue;
                // if the board blocks the hand
                if(utils::handBlockedByBoard(opponentHand, board)) continue;
                counterfactualValue += rangeOpponent[opponentHandIdx].item<float>();
                count[handIdx][opponentHandIdx] += 1;
            }
            equity[handIdx] = counterfactualValue;
        }
        return {equity, count};
    }

};