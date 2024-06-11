#pragma once

#include "../utils/card.h"
#include "../settings.h"

#include <cassert>
#include <vector>

enum class HandStrengthType{
    HighCard = 0,
    Pair = 1,
    TwoPair = 2,
    ThreeOfAKind = 3,
    FourOfAKind = 4,
};

struct HandStrength{
    HandStrengthType type;
    std::vector<int> typeRanks;
    std::vector<int> kickerRanks;
};

namespace handComp{
    // returns a flag, 1 if hand1 is better, 0 if they are equal, -1 if hand2 is better
    int compareHands(const std::vector<Card> & board, const std::vector<Card> & hand1, const std::vector<Card> & hand2);
    HandStrength strength(const std::vector<Card> & board, const std::vector<Card> & hand);
}