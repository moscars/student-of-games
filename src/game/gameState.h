#pragma once

#include "../utils/card.h"
#include "../utils/utils.h"

#include <array>

#include <torch/torch.h>

struct GameState{
    std::array<int, 2> bets;
    int street;
    std::vector<Card> board;
    Player actingPlayer;
    bool isStartOfGame;
    bool isTerminal;
    ActionType actionToThisState;
    size_t actionIdxToThisState;
};

// print
std::ostream & operator<<(std::ostream & os, const GameState & state);