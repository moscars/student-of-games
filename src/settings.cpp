#include "settings.h"

namespace settings{

    // List of pot-scaled bet sizes to use in tree
    const std::vector<float> betSizing = {1.0f};

    const std::vector<std::string> deckCards = {"Ah", "As", "Ad", "Kh", "Ks", "Kd", "Qh", "Qs", "Qd", "Jh", "Js", "Jd"};
    const int deckSize = 12;
    const int handSize = 1;
    const int numPossibleHands = 12; // deckSize choose handSize

    // deckSize + nextToAct + betsMade + ranges;
    const int networkInputSize = deckSize + 3 + 2 + numPossibleHands * 2; 
    const int networkOutputSize = 2 * numPossibleHands;
    const int hiddenSize = 300;

    std::vector<std::vector<Card>> generateAllHands(){
        std::vector<std::vector<Card>> hands;

        if(handSize == 1){
            for(int i = 0; i < static_cast<int>(deckCards.size()); i++){
                hands.push_back({Card(deckCards[i])});
            }
        } else if (handSize == 2){
            for(int i = 0; i < static_cast<int>(deckCards.size()); i++){
                for(int j = i + 1; j < static_cast<int>(deckCards.size()); j++){
                    hands.push_back({Card(deckCards[i]), Card(deckCards[j])});
                }
            }
        } else if(handSize == 3){
            for(int i = 0; i < static_cast<int>(deckCards.size()); i++){
                for(int j = i + 1; j < static_cast<int>(deckCards.size()); j++){
                    for(int k = j + 1; k < static_cast<int>(deckCards.size()); k++){
                        hands.push_back({Card(deckCards[i]), Card(deckCards[j]), Card(deckCards[k])});
                    }
                }
            }
        }

        // make sure we get the correct number of hands
        assert(static_cast<int>(hands.size()) == numPossibleHands);
        return hands;
    }


    const std::vector<std::vector<Card>> possibleHands = generateAllHands();
    
    const int stackSize = 5000;
    const int ante = 100;

    const int warmupCFRIter = 100;
    const int CFRIter = 200;

    const int numStreets = 3;

    int cnt = 0;

    const bool useCuda = false;
    const bool useMPS = true;

    const float regretEpsilon = 1.0f/1000000000.0f;

    // for 500 -> 0.00412 is good and scaling is linear with iters
    // 35000 per hand / 300 * 50 hands ish 5000
    const float selfPlaySavingThreshold = 0.0015f * 0.35f;
    const float selfPlayEpsilon = 0.1f;
    
    const int maxDepth = 3;

    bool selfPlay = true;
    std::string modelName = "model.pt";
    std::string fileName = "soundSelfPlay.txt";

    bool solving = false;

    std::mutex saveMutex;
    std::shared_mutex cacheMutex;
    std::shared_mutex foldCacheMutex;
    std::mutex printMutex;
};