#include "netHelper.h"

namespace netHelper{
    std::unordered_map<int, torch::Tensor> maskCache;
    std::shared_mutex maskCacheMutex;

    torch::Tensor getBoardMask(torch::Tensor boardTen, int bitBoard){
        
        {
            std::shared_lock<std::shared_mutex> lock(maskCacheMutex);
            if(maskCache.count(bitBoard)){
                return maskCache[bitBoard];
            }
        }


        torch::Tensor mask = torch::ones({settings::numPossibleHands});
        for(int i = 0; i < settings::numPossibleHands; i++){
            std::vector<Card> hand = settings::possibleHands[i];
            std::vector<Card> board;

            for(int j = 0; j < settings::deckSize; j++){
                if(boardTen[j].item<int>() == 1){
                    board.push_back(settings::deckCards[j]);
                }
            }
            
            if(utils::handBlockedByBoard(hand, board)){
                mask[i] = 0;
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(maskCacheMutex);
            maskCache[bitBoard] = mask.clone();
        }

        return mask;
    }

    torch::Tensor customGELU(torch::Tensor x){
        return x * 0.5f * (1.0f + torch::erf(x / 1.41421356237f));
    }
};