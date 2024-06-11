#pragma once

#include "utils/card.h"

#include <vector>
#include <string>
#include <cassert>
#include <mutex>
#include <shared_mutex>

namespace settings{
    // List of pot-scaled bet sizes to use in tree
    extern const std::vector<float> betSizing;
    // extern const std::vector<std::string> possibleHands;
    extern const std::vector<std::string> deckCards;
    extern const int deckSize;

    extern const std::vector<std::vector<Card>> possibleHands;
    extern const int numPossibleHands;
    extern const int handSize;
    extern const int networkInputSize;
    extern const int networkOutputSize;
    extern const int hiddenSize;

    extern const int stackSize;
    extern const int ante;

    extern const int warmupCFRIter;
    extern const int CFRIter;
    
    extern const int maxDepth;

    extern const float regretEpsilon;

    extern const float selfPlaySavingThreshold;
    
    extern const float selfPlayEpsilon;

    extern const int numStreets;

    extern int cnt;

    extern const bool useCuda;
    extern const bool useMPS;

    extern bool selfPlay;
    extern std::string modelName;
    extern std::string fileName;

    extern bool solving;

    extern std::mutex saveMutex;
    extern std::shared_mutex cacheMutex;
    extern std::shared_mutex foldCacheMutex;
    extern std::mutex printMutex;

    std::vector<std::vector<Card>> generateAllHands();
};