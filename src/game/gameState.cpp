#include "gameState.h"

std::ostream & operator<<(std::ostream & os, const GameState & state){

    os << state.actionToThisState << " -> ";
    os << "Bets: " << state.bets[0] << " " << state.bets[1] << " ";
    os << "Board: ";
    for(auto card : state.board){
        os << card << " ";
    }
    return os;
}
