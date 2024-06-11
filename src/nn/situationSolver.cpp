#include "situationSolver.h"

SituationSolver::SituationSolver(std::shared_ptr<Net> model_, const std::string & filename_){
    model = model_;
    filename = filename_;
    numCores = 16;
}

std::vector<torch::Tensor> SituationSolver::solveLines(const std::vector<std::string> & lines){

    std::vector<torch::Tensor> targetValues;

    int index = 0;
    for(const std::string & line : lines){
        std::vector<std::string> parts = utils::split(line, '|');
        std::vector<Card> boardCards;
        for(int i = 0; i < settings::deckSize; i++){
            if(parts[i] == "1"){
                boardCards.push_back(Card(settings::deckCards[i]));
            }
        }

        Player nextToAct = Player::P1;
        
        assert(parts[settings::deckSize] == "1" || parts[settings::deckSize + 1] == "1" || parts[settings::deckSize + 2] == "1");
        int total = (parts[settings::deckSize] == "1") + (parts[settings::deckSize + 1] == "1") + (parts[settings::deckSize + 2] == "1");
        assert(total == 1);

        if(parts[settings::deckSize + 1] == "1"){
            nextToAct = Player::P2;
        } else if(parts[settings::deckSize + 2] == "1"){
            nextToAct = Player::CHANCE;
        }

        int bet1 = static_cast<int>(std::round(std::stod(parts[settings::deckSize + 3]) * static_cast<float>(settings::stackSize)));
        int bet2 = static_cast<int>(std::round(std::stod(parts[settings::deckSize + 4]) * static_cast<float>(settings::stackSize)));

        assert(bet1 > 0);
        assert(bet2 > 0);

        torch::Tensor range1 = torch::zeros({settings::numPossibleHands});
        torch::Tensor range2 = torch::zeros({settings::numPossibleHands});

        for(int i = 0; i < settings::numPossibleHands; i++){
            range1[i] = std::stod(parts[i + settings::deckSize + 5]);
            range2[i] = std::stod(parts[i + settings::deckSize + 5 + settings::numPossibleHands]);
        }

        DeepResolving resolving;
        std::shared_ptr<PublicBeliefState> root = std::make_shared<PublicBeliefState>();
        root->setBoard(boardCards);
        root->setStreet(static_cast<int>(boardCards.size()) + 1);
        root->setPlayer(nextToAct);
        root->setBets({bet1, bet2});
        
        resolving.resolveStartOfGame(root, range1, range2);

        torch::Tensor target = resolving.getRootCFV();
        target = target.view({2 * settings::numPossibleHands});
        
        assert(root->getPotSize() > 0);
        assert(!torch::any(torch::isnan(target)).item<bool>());

        target /= root->getPotSize();
        targetValues.push_back(target);

        if(index % 100 == 0){
            ppr("Solved situations:", index);
        }
        index++;
    }

    return targetValues;
}

void SituationSolver::solve(){
    std::ifstream stream(filename);
    if(!stream.is_open()){
        throw std::runtime_error("Could not open file to solve situations");
    }

    std::string line;
    std::vector<torch::Tensor> targetValues;

    std::vector<std::string> lines;
    while(std::getline(stream, line)){
        lines.push_back(line);
    }

    int totalLines = static_cast<int>(lines.size());
    int linesPerCore = totalLines / numCores;

    ppr("Total cores:", numCores);
    ppr("Total lines:", totalLines);
    ppr("Lines per core:", linesPerCore);

    // split into work for each core
    std::vector<std::vector<std::string>> linesForCores(numCores);
    for(int i = 0; i < numCores; i++){
        int start = i * linesPerCore;
        int end = (i + 1) * linesPerCore;
        if(i == numCores - 1){
            end = totalLines;
        }
        linesForCores[i] = std::vector<std::string>(lines.begin() + start, lines.begin() + end);
    }

    // start threads for each core (using futures)
    settings::solving = true;
    std::vector<std::future<std::vector<torch::Tensor>>> futures;
    for(int i = 0; i < numCores; i++){
        ppr("Starting core", i);
        futures.push_back(std::async(std::launch::async, &SituationSolver::solveLines, this, linesForCores[i]));
    }


    // get results from each core
    for(int i = 0; i < numCores; i++){
        std::vector<torch::Tensor> coreResults = futures[i].get();
        ppr("Got result from core", i);
        for(torch::Tensor result : coreResults){
            targetValues.push_back(result);
        }
    }

    torch::Tensor targets = torch::stack(targetValues);
    pr(targets.sizes());
    torch::save(targets, filename + ".targets");
    pr("Finshed saving to file");
    settings::solving = false;
}