#include "handComp.h"

namespace handComp{


HandStrength strength(const std::vector<Card> & board, const std::vector<Card> & hand){
    std::vector<Card> allCards = board;
    allCards.insert(allCards.end(), hand.begin(), hand.end());
    std::array<int, 13> rankCount = {0};
    for(const Card & card : allCards){
        assert(card.getRank() >= 0 && card.getRank() < 13);
        rankCount[card.getRank()]++;
    }


    // check for four of a kind
    for(int i = 0; i < 13; i++){
        if(rankCount[i] == 4){
            HandStrength handStrength;
            handStrength.type = HandStrengthType::FourOfAKind;
            handStrength.typeRanks = {i};
            return handStrength;
        }
    }

    std::vector<int> threeOfAKindRanks;
    std::vector<int> pairRanks;
    for(int i = 0; i < 13; i++){
        if(rankCount[i] == 3){
            threeOfAKindRanks.push_back(i);
        }
        if(rankCount[i] == 2){
            pairRanks.push_back(i);
        }
    }

    // check for three of a kind
    if(threeOfAKindRanks.size() >= 1){
        // find maximum three of a kind
        int maxThreeOfAKind = threeOfAKindRanks[0];
        if(threeOfAKindRanks.size() == 2){
            maxThreeOfAKind = std::max(threeOfAKindRanks[0], threeOfAKindRanks[1]);
        }

        HandStrength handStrength;
        handStrength.type = HandStrengthType::ThreeOfAKind;
        handStrength.typeRanks = {maxThreeOfAKind};
        int kicker = -1;
        for(int i = 0; i < 13; i++){
            if(rankCount[i] >= 1 && i != maxThreeOfAKind){
                kicker = i;
            }
        }
        handStrength.kickerRanks = {kicker};
        // pop off until two kickers
        return handStrength;
    }

    // check for two pair
    if(pairRanks.size() >= 2){
        HandStrength handStrength;
        handStrength.type = HandStrengthType::TwoPair;
        handStrength.typeRanks = {pairRanks[0], pairRanks[1]};
        return handStrength;
    }

    // check for pair
    if(pairRanks.size() == 1){
        HandStrength handStrength;
        handStrength.type = HandStrengthType::Pair;
        handStrength.typeRanks = {pairRanks[0]};
        std::vector<int> kickers;
        for(int i = 0; i < 13; i++){
            if(rankCount[i] >= 1 && i != pairRanks[0]){
                kickers.push_back(i);
            }
        }
        // sort kickers
        std::sort(kickers.begin(), kickers.end(), std::greater<int>());
        // pop off until two kickers
        if(kickers.size() >= 2){
            handStrength.kickerRanks = {kickers[0], kickers[1]};
        } else{
            handStrength.kickerRanks = kickers;
        }
        return handStrength;
    }

    // check for high card
    HandStrength handStrength;
    handStrength.type = HandStrengthType::HighCard;
    std::vector<int> kickers;
    for(int i = 0; i < 13; i++){
        if(rankCount[i] >= 1){
            kickers.push_back(i);
        }
    }
    // sort kickers
    std::sort(kickers.begin(), kickers.end(), std::greater<int>());
    // pop off until four kickers
    if(kickers.size() >= 4){
        handStrength.kickerRanks = {kickers[0], kickers[1], kickers[2], kickers[3]};
    } else{
        handStrength.kickerRanks = kickers;
    }

    return handStrength;
}

int compareHands(const std::vector<Card> & board, const std::vector<Card> & hand1, const std::vector<Card> & hand2){
    HandStrength handStrength1 = strength(board, hand1);
    HandStrength handStrength2 = strength(board, hand2);

    if(handStrength1.type > handStrength2.type){
        return 1;
    } else if(handStrength1.type < handStrength2.type){
        return -1;
    }

    // compare type ranks
    for(size_t i = 0; i < handStrength1.typeRanks.size(); i++){
        if(handStrength1.typeRanks[i] > handStrength2.typeRanks[i]){
            return 1;
        } else if(handStrength1.typeRanks[i] < handStrength2.typeRanks[i]){
            return -1;
        }
    }

    // compare kicker ranks
    for(size_t i = 0; i < handStrength1.kickerRanks.size(); i++){
        if(handStrength1.kickerRanks[i] > handStrength2.kickerRanks[i]){
            return 1;
        } else if(handStrength1.kickerRanks[i] < handStrength2.kickerRanks[i]){
            return -1;
        }
    }

    return 0;
}


}