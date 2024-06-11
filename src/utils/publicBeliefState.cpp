#include "publicBeliefState.h"

std::pair<std::string, std::string> PublicBeliefState::nodeToGraphViz() {
    std::string graphVizNode;

    // Start the graphViz representation for the node
    std::string name = std::to_string(reinterpret_cast<uintptr_t>(this));
    graphVizNode += "\"" + std::to_string(reinterpret_cast<uintptr_t>(this)) + "\" [";

    // Label
    graphVizNode += "label=\"";

    // Add node details

    std::string node_type_str = "";
    if(actionToThisNode == ActionType::FOLD){
        node_type_str = "FOLD";
    } else if(actionToThisNode == ActionType::CALL){
        node_type_str = "CALL";
    } else if(actionToThisNode == ActionType::CHECK || actionToThisNode == ActionType::TCHECK){
        node_type_str = "CHECK";
    } else if(actionToThisNode == ActionType::BET){
        node_type_str = "BET";
    } else{
        node_type_str = "unknown";
    }
    
    graphVizNode += "Current player: " + std::to_string(static_cast<int>(player) + 1);
    graphVizNode += "\\n" + node_type_str;

    std::string boardString = "";
    for(auto card : board){
        boardString += card.toString();
    }

    graphVizNode += "\\nStreet: " + std::to_string(street);
    graphVizNode += "\\nBoard: " + boardString;
    graphVizNode += "\\nBets: " + std::to_string(bets[0]) + " " + std::to_string(bets[1]);
    graphVizNode += "\\nTerminal: " + std::string(terminal ? "true" : "false");
    graphVizNode += "\\nDepth: " + std::to_string(depth);
    graphVizNode += "\\nExploit: " + std::to_string(exploitability);
    //print the strategy

    if(strategy.numel() > 0){
        graphVizNode += "\\nStrategy: ";
        for(int currPlayer = 0; currPlayer < strategy.size(0); currPlayer++){
            for(int action = 0; action < strategy.size(1); action++){
                graphVizNode += std::to_string(strategy[currPlayer][action].item<float>()) + " ";
            }
            graphVizNode += "\\n";
        }
    }

    //print the ranges
    graphVizNode += "\\nRanges: ";
    for(int currPlayer = 0; currPlayer < ranges.size(0); currPlayer++){
        for(int card = 0; card < ranges.size(1); card++){
            graphVizNode += std::to_string(ranges[currPlayer][card].item<float>()) + " ";
        }
        graphVizNode += "\\n";
    }

    //print the values
    graphVizNode += "\\nValues: ";
    for(int currPlayer = 0; currPlayer < cfValues.size(0); currPlayer++){
        for(int card = 0; card < cfValues.size(1); card++){
            graphVizNode += std::to_string(cfValues[currPlayer][card].item<float>()) + " ";
        }
        graphVizNode += "\\n";
    }

    graphVizNode += "\"";

    // Node shape and other attributes can be set here
    graphVizNode += ", shape=box";

    // End the graphViz representation for the node
    graphVizNode += "];\n";

    return {name, graphVizNode};
}

void PublicBeliefState::printImp() const {
    std::cout << "Player: " << static_cast<int>(getPlayer()) << std::endl;
    std::cout << "Bets: " << getBets()[0] << " " << getBets()[1] << std::endl;
    std::cout << "Board: ";
    for(auto card : getBoard()){
        std::cout << card << " ";
    }

    std::cout << std::endl;
}


std::ostream& operator<<(std::ostream& os, const PublicBeliefState& pbs){
    os << "Player: " << static_cast<int>(pbs.getPlayer()) << std::endl;
    os << "Action: " << static_cast<int>(pbs.getActionToThisNode()) << std::endl;
    os << "Terminal: " << pbs.isTerminal() << std::endl;
    os << "Depth: " << pbs.getDepth() << std::endl;
    os << "Street: " << pbs.getStreet() << std::endl;
    os << "Bets: " << pbs.getBets()[0] << " " << pbs.getBets()[1] << std::endl;
    os << "Board: ";
    for(auto card : pbs.getBoard()){
        os << card << " ";
    }

    os << "\nRanges\n";
    for(int player = 0; player < pbs.ranges.size(0); player++){
        for(int card = 0; card < pbs.ranges.size(1); card++){
            os << pbs.ranges[player][card].item<float>() << " ";
        }
        os << std::endl;
    }

    os << std::endl;

    return os;
}
