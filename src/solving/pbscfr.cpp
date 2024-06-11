#include "pbscfr.h"

void PBSCFR::runCFR(std::shared_ptr<PublicBeliefState> root){
    root->ranges = torch::ones({2, settings::deckSize}) / settings::deckSize;
    for(int i = 1; i <= settings::CFRIter; i++){
        if(i % 50 == 0) ppr("CFR iter:", i);
        cfrDFS(root, i);
    }

    normalizeStrat(root);
}

void PBSCFR::cfrIter(std::shared_ptr<PublicBeliefState> node, torch::Tensor opponentRange, int iter){
    int opponentIndex = static_cast<int>(utils::otherPlayer(node->getPlayer()));
    node->ranges[opponentIndex] = opponentRange;
    cfrDFS(node, iter);
}


void PBSCFR::cfrDFS(std::shared_ptr<PublicBeliefState> node, int iter){

    if(node->isTerminal()){

        torch::Tensor values = torch::zeros_like(node->ranges);
        if(node->getActionToThisNode() == ActionType::FOLD){
            auto valueP1 = equity::getTerminalEquityFold(node->getBoard(), node->ranges[1]).first;
            auto valueP2 = equity::getTerminalEquityFold(node->getBoard(), node->ranges[0]).first;
            values = torch::stack({valueP1, valueP2});
            Player currentPlayer = node->getPlayer();
            assert(currentPlayer != Player::CHANCE);
            int opponent_index = static_cast<int>(utils::otherPlayer(currentPlayer));
            values[opponent_index] *= -1;
        } else{
            values = equity::getTerminalEquityCall(node->getBoard(), node->ranges[0], node->ranges[1]).first;
        }

        values *= node->getPotSize();
        node->cfValues = values.clone();
    } else {

        if(node->getDepth() == settings::maxDepth){
            assert(node->getBoard().size() == 1);
            assert(!settings::solving);
            assert(node->getPlayer() != Player::CHANCE);
        }

        int actions_count = static_cast<int>(node->getChildren().size());
        torch::Tensor current_strategy;

        if(node->getPlayer() == Player::CHANCE){
            current_strategy = node->strategy.clone();
        } else{
            
            // REGRET MATCHING
            if(node->regrets.numel() == 0){
                node->regrets = torch::full({settings::numPossibleHands, actions_count}, regret_epsilon);
            }

            torch::Tensor positiveRegrets = node->regrets.clone();
            positiveRegrets = torch::where(positiveRegrets > regret_epsilon, positiveRegrets, regret_epsilon);

            torch::Tensor regretSum = positiveRegrets.sum(1); // CxA -> C -> The total regret for each infoset

            current_strategy = positiveRegrets.clone();
            // normalize the strategy (each row -> strategy for one infoset -> should sum to one)
            current_strategy /= regretSum.view({settings::numPossibleHands, 1});
        }

        if(node->getPlayer() == Player::CHANCE){
            int p1 = static_cast<int>(Player::P1);
            int p2 = static_cast<int>(Player::P2);

            for(size_t i = 0; i < node->getChildren().size(); i++){
                std::shared_ptr<PublicBeliefState> child = node->getChildren()[i];

                if(child->ranges.numel() == 0) child->ranges = torch::zeros({2, settings::deckSize});

                child->ranges[p1] = node->ranges[p1] * current_strategy.select(1, i);
                child->ranges[p2] = node->ranges[p2] * current_strategy.select(1, i);
            }

        } else {
            int currentPlayerIndex = static_cast<int>(node->getPlayer());
            int opponentIndex = static_cast<int>(utils::otherPlayer(node->getPlayer()));

            for(size_t i = 0; i < node->getChildren().size(); i++){
                std::shared_ptr<PublicBeliefState> child = node->getChildren()[i];

                if(child->ranges.numel() == 0) child->ranges = torch::zeros({2, settings::deckSize});

                child->ranges[currentPlayerIndex] = node->ranges[currentPlayerIndex] * current_strategy.select(1, i);
                child->ranges[opponentIndex] = node->ranges[opponentIndex].clone();
            }
        }

        std::vector<torch::Tensor> childrenCFVs;

        for(std::shared_ptr<PublicBeliefState> child : node->getChildren()){
            cfrDFS(child, iter);
            childrenCFVs.push_back(child->cfValues);
        }

        node->cfValues = torch::zeros({2, settings::numPossibleHands});

        if(node->getPlayer() != Player::CHANCE){
            int currentPlayerIndex = static_cast<int>(node->getPlayer());
            int opponentIndex = static_cast<int>(utils::otherPlayer(node->getPlayer()));

            for(size_t childIdx = 0; childIdx < node->getChildren().size(); childIdx++){
                std::shared_ptr<PublicBeliefState> child = node->getChildren()[childIdx];
                node->cfValues[currentPlayerIndex] += child->cfValues[currentPlayerIndex] * current_strategy.select(1, childIdx); // The strategy given that we played this action
                node->cfValues[opponentIndex] += child->cfValues[opponentIndex];
            }
        } else {
            int p1 = static_cast<int>(Player::P1);
            int p2 = static_cast<int>(Player::P2);
            for(std::shared_ptr<PublicBeliefState> child : node->getChildren()){
                node->cfValues[p1] += child->cfValues[p1];
                node->cfValues[p2] += child->cfValues[p2];
            }
        }

        if(node->getPlayer() != Player::CHANCE){
            int player = static_cast<int>(node->getPlayer());
            // Update regrets
            torch::Tensor current_regrets = torch::zeros({actions_count, settings::numPossibleHands});
            for(size_t childIdx = 0; childIdx < node->getChildren().size(); childIdx++){
                std::shared_ptr<PublicBeliefState> child = node->getChildren()[childIdx];
                // the regret for each infoset given that I took action A is the 
                current_regrets[childIdx] = child->cfValues[player] - node->cfValues[player];
            }

            current_regrets = current_regrets.t();

            // Sum cumulative regrets
            node->regrets += current_regrets;
            node->regrets = torch::where(node->regrets > regret_epsilon, node->regrets, regret_epsilon);

            updateAverageStrategy(node, current_strategy, iter);
        }
    }
}

void PBSCFR::updateAverageStrategy(std::shared_ptr<PublicBeliefState> node, torch::Tensor current_strategy, int iter){
    // std::array<int, 2> vals = {100, 300};
    // if(node->getBets() == vals){
    //     pr(current_strategy);
    // }

    if(iter <= settings::warmupCFRIter) return;
    // if(node->strategy.numel() == 0){
    //     node->strategy = torch::zeros({settings::numPossibleHands, static_cast<int>(node->getChildren().size())});
    // }
    // node->strategy += current_strategy;
    node->strategiesAtCFR.push_back(current_strategy.clone());

    // if(node->iter_weight_sum.numel() == 0){
    //     node->iter_weight_sum = torch::zeros({static_cast<int>(settings::deckSize)});
    // }
    // int player = static_cast<int>(node->getPlayer());
    // torch::Tensor iter_weight_contribution = node->ranges[player].clone();
    // iter_weight_contribution = torch::where(iter_weight_contribution > 0, iter_weight_contribution, regret_epsilon);
    // node->iter_weight_sum += iter_weight_contribution;

    // torch::Tensor iter_weight = iter_weight_contribution / node->iter_weight_sum;
    // torch::Tensor expand_weight = iter_weight.view({static_cast<int>(settings::deckSize), 1});
    // torch::Tensor old_strategy_scale = 1 - expand_weight;
    // node->strategy *= old_strategy_scale;
    // torch::Tensor strategy_addition = current_strategy * expand_weight;

    // node->strategy += strategy_addition;
}

void PBSCFR::normalizeStrat(std::shared_ptr<PublicBeliefState> node){
    if(node->isTerminal()) return;

    if(node->getPlayer() != Player::CHANCE){
        node->strategy = torch::zeros({settings::deckSize, static_cast<int>(node->getChildren().size())});
        for(int i = 0; i < static_cast<int>(node->strategiesAtCFR.size()); i++){
            node->strategy += node->strategiesAtCFR[i] * (i + 1);
        }

        // node->strategy /= (settings::CFRIter - settings::warmupCFRIter);
        // normalize strategy
        for(int i = 0; i < settings::deckSize; i++){
            node->strategy[i] /= torch::sum(node->strategy[i]);
        }

    }
    
    for(auto child : node->getChildren()){
        normalizeStrat(child);
    }
}
