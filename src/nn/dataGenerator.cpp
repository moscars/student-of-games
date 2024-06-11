#include "dataGenerator.h"

void DataGenerator::generate(const std::string & filename){
    // we need to generate random bet sizes
    // random ranges
    // and a random board
    // then generate a random valid player for the next action


    // generate random ranges
    std::vector<float> rangeP1(6, 0.0f);
    std::vector<float> rangeP2(6, 0.0f);

    generateRange(rangeP1, 1.0f, 0, 6);
    generateRange(rangeP2, 1.0f, 0, 6);

    // generate random board
    int board = torch::randint(0, 6, {1}).item<int>();
    std::vector<float> deck(settings::deckSize, 0.0f);
    deck[board] = 1.0f;

    // normalize ranges (and remove board)
    rangeP1[board] = 0.0f;
    rangeP2[board] = 0.0f;
    float sumP1 = std::accumulate(rangeP1.begin(), rangeP1.end(), 0.0f);
    float sumP2 = std::accumulate(rangeP2.begin(), rangeP2.end(), 0.0f);

    // generate scaling of ranges
    for(float & val : rangeP1) val /= sumP1;
    for(float & val : rangeP2) val /= sumP2;

    // generate random bets
    float betP1 = torch::rand({1}).item<float>();// * settings::stackSize;
    float betP2 = torch::rand({1}).item<float>();// * settings::stackSize;

    while(betP1 * static_cast<float>(settings::stackSize) < settings::ante){
        betP1 = torch::rand({1}).item<float>();
    }
    while(betP2 * static_cast<float>(settings::stackSize) < settings::ante){
        betP2 = torch::rand({1}).item<float>();
    }

    std::vector<float> actingPlayer(3, 0.0f);

    // with a low probability set the bets equal
    float randNum = torch::rand({1}).item<float>();
    if(randNum < 0.1){
        betP2 = betP1;
        // acting player is p1
        actingPlayer[0] = 1.0f;
    }

    if(betP1 > betP2){
        // p2 to act
        actingPlayer[1] = 1.0f;
    } else {
        // p1 to act
        actingPlayer[0] = 1.0f;
    }

    saveData(deck, actingPlayer, betP1, betP2, rangeP1, rangeP2, filename);
}

void DataGenerator::saveData(const std::vector<float> & board, 
                            const std::vector<float> & actingPlayer, 
                            float betP1, float betP2, 
                            const std::vector<float> & rangeP1, 
                            const std::vector<float> & rangeP2,
                            const std::string & filename){

    std::ofstream file("data/" + filename, std::ios::app);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file to save data");
    }

    for(float val : board){
        file << val << "|";
    }

    for(float val : actingPlayer){
        file << val << "|";
    }

    file << betP1 << "|";
    file << betP2 << "|";

    for(float val : rangeP1){
        file << val << "|";
    }

    for(float val : rangeP2){
        file << val << "|";
    }
    file << std::endl;
}



void DataGenerator::generateRange(std::vector<float> & range, float mass, size_t left, size_t right){
    assert(left < right);
    assert(mass <= 1.0f);
    assert(mass >= 0.0f);
    assert(left >= 0);
    assert(right <= range.size());

    if(left == right - 1){
        range[left] = mass;
        return;
    }

    float randNum = torch::rand({1}).item<float>();
    float leftMass = randNum * mass;
    float rightMass = mass - leftMass;

    int cards = right - left;
    int middle = (left + right) / 2;

    if(cards % 2 == 1){
        // randomize which way the random number goes
        float rand2 = torch::rand({1}).item<float>();
        if(rand2 < 0.5){
            middle++;
        }
    }

    generateRange(range, leftMass, left, middle);
    generateRange(range, rightMass, middle, right);
}