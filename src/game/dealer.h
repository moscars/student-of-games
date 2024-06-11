#pragma once

#include "../solving/continualResolving.h"
#include "../settings.h"

#include <vector>
#include <algorithm>
#include <random>

struct GameResults{
    std::vector<float> IOResults;
    std::vector<float> aivatResults;
    std::vector<float> FoldAndCallResults;
    std::vector<float> FoldAndRaiseResultsWithAIVAT;
};

class Dealer{

public:
    Dealer(bool trackResults_, Player playerToTrack_);
    Dealer();
    void playHand();
    GameResults getResults() const;

private:
    std::vector<Card> deck;
    std::vector<float> IOResults;
    std::vector<float> aivatResults;
    std::vector<float> FoldAndCallResults;
    std::vector<float> FoldAndRaiseResultsWithAIVAT;
    bool trackResults;
    Player playerToTrack;
};
