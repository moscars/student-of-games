#include "inference.h"

torch::Tensor Inference::evaluate(std::shared_ptr<PublicBeliefState> node, torch::Tensor player1Range, torch::Tensor player2Range, std::optional<int> modelIndex){
    assert(!node->isTerminal());

    torch::Tensor board = torch::zeros({settings::deckSize});
    int bitBoard = utils::encodeBoard(node->getBoard());
    for(Card card : node->getBoard()){
        int index = -1;
        for(int i = 0; i < settings::deckSize; i++){
            if(Card(settings::deckCards[i]) == card){
                index = i;
                break;
            }
        }
        assert(index != -1);
        board[index] = 1;
    }

    torch::Tensor actingPlayer = torch::zeros({3});
    actingPlayer[static_cast<int>(node->getPlayer())] = 1;

    torch::Tensor bets = torch::zeros({2});
    bets[0] = static_cast<float>(node->getBets()[0]) / static_cast<float>(settings::stackSize);
    bets[1] = static_cast<float>(node->getBets()[1]) / static_cast<float>(settings::stackSize);

    assert(bets[0].item<float>() > 0);
    assert(bets[1].item<float>() > 0);

    auto r0Sum = torch::sum(player1Range);
    auto r1Sum = torch::sum(player2Range);
    auto normR0 = player1Range / r0Sum;
    auto normR1 = player2Range / r1Sum;

    torch::Tensor input = torch::cat({board, actingPlayer, bets, normR0, normR1});

    torch::NoGradGuard noGrad;

    input = input.view({1, -1});

    torch::Tensor output;

    if(!settings::selfPlay){
        // assert(modelIndex.has_value());
    }

    if(modelIndex.has_value()){
        if(modelIndex.value() == 0){
            assert(netHolder::modelP1 != nullptr);
            (*netHolder::modelP1)->eval();
            output = (*netHolder::modelP1)->forward(input);
        } else{
            assert(modelIndex.value() == 1);
            assert(netHolder::modelP2 != nullptr);
            (*netHolder::modelP2)->eval();
            output = (*netHolder::modelP2)->forward(input);
        }
    } else{
        assert(netHolder::model != nullptr);
        (*netHolder::model)->eval();
        output = (*netHolder::model)->forward(input);
    }

    input = input.view({-1});

    if(settings::selfPlay){
        float randomNumber = torch::rand({1}).item<float>();
        if(randomNumber < settings::selfPlaySavingThreshold){
            if(settings::solving){
                if(randomNumber < settings::selfPlaySavingThreshold / 10){
                    dataSaver::saveData(input, true);
                }
            } else{
                dataSaver::saveData(input, false);
            }
        }
    }

    output = output.view({2, settings::numPossibleHands});

    auto rangeBoardMask = netHelper::getBoardMask(board, bitBoard);
    // mask out impossible board
    output[0] = output[0] * rangeBoardMask;
    output[1] = output[1] * rangeBoardMask;

    output[0] *= r1Sum;
    output[1] *= r0Sum;

    return output;
}