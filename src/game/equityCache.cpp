#include "equityCache.h"

torch::Tensor EquityCache::getTerminalEquityCall(const std::vector<Card> & board, torch::Tensor rangeP1, torch::Tensor rangeP2){
    int bitBoard = utils::encodeBoard(board);

    bool cacheHit = false;
    
    {
        std::shared_lock<std::shared_mutex> lock(settings::cacheMutex);
        cacheHit = cache.count(bitBoard);
    }

    if(cacheHit){
        torch::Tensor winners;

        {
            std::shared_lock<std::shared_mutex> lock(settings::cacheMutex);
            winners = cache[bitBoard];
        }
        
        torch::Tensor winnersT = winners.transpose(0, 1);
        // torch::Tensor equity = torch::zeros({2, settings::numPossibleHands});
        torch::Tensor equity2 = torch::zeros({2, settings::numPossibleHands});

        // for(int p1 = 0; p1 < settings::numPossibleHands; p1++){
        //     for(int p2 = 0; p2 < settings::numPossibleHands; p2++){
        //         equity[0][p1] += winners[p1][p2] * rangeP2[p2];
        //         equity[1][p2] -= winners[p1][p2] * rangeP1[p1];

        //         equity[0][p1] -= winners[p2][p1] * rangeP2[p2];
        //         equity[1][p2] += winners[p2][p1] * rangeP1[p1];
        //     }
        // }

        for(int hand = 0; hand < settings::numPossibleHands; hand++){
            equity2[0][hand] += torch::dot(winners[hand], rangeP2);
            equity2[0][hand] -= torch::dot(winnersT[hand], rangeP2);

            equity2[1][hand] += torch::dot(winners[hand], rangeP1);
            equity2[1][hand] -= torch::dot(winnersT[hand], rangeP1);
        }

        // assert(torch::allclose(equity, equity2, 100, 1e-5));

        float scale = 1 / static_cast<float>(settings::deckSize - 2 * settings::handSize);

        if(board.size() == 0){
            auto ten = equity2 * scale * scale / 2;

            // auto [eq, cnt] = equity::getTerminalEquityCall(board, rangeP1, rangeP2);
            // assert(torch::allclose(ten, eq, 1e-5, 1e-5));

            return ten;
        } else if(board.size() == 1){
            auto ten = equity2 * scale / 2;

            // auto [eq, cnt] = equity::getTerminalEquityCall(board, rangeP1, rangeP2);
            // assert(torch::allclose(ten, eq, 1e-5, 1e-5));

            return ten;
        } else if(board.size() == 2){
            auto ten = equity2 / 2;

            // auto [eq, cnt] = equity::getTerminalEquityCall(board, rangeP1, rangeP2);
            // assert(torch::allclose(ten, eq, 1e-5, 1e-5));

            return ten;
        } else{
            assert(false);
        }
    } else{

        auto [equity, count] = equity::getTerminalEquityCall(board, rangeP1, rangeP2);

        std::unique_lock<std::shared_mutex> savingLock(settings::cacheMutex);
        cache[bitBoard] = count;
        return equity;
    }

}

torch::Tensor EquityCache::getTerminalEquityFold(const std::vector<Card> & board, torch::Tensor opponentRange){
    int bitBoard = utils::encodeBoard(board);

    bool cacheHit = false;

    {
        std::shared_lock<std::shared_mutex> lock(settings::foldCacheMutex);
        cacheHit = foldCache.count(bitBoard);
    }

    if(cacheHit){
        torch::Tensor winners;
        {
            std::shared_lock<std::shared_mutex> lock(settings::foldCacheMutex);
            winners = foldCache[bitBoard];
        }

        // torch::Tensor equity2 = torch::zeros({settings::numPossibleHands});
        
        // for(int hand = 0; hand < settings::numPossibleHands; hand++){
        //     equity2[hand] = torch::dot(winners[hand], opponentRange);
        // }
        torch::Tensor equity = torch::matmul(winners, opponentRange);

        // auto equity3 = equity::getTerminalEquityFold(board, opponentRange).first;
        // assert(torch::allclose(equity, equity3));

        return equity;

    } else{
        auto [eq, count] = equity::getTerminalEquityFold(board, opponentRange);

        {
            std::unique_lock<std::shared_mutex> savingLock(settings::foldCacheMutex);
            foldCache[bitBoard] = count;
        }

        return eq;
    }
}