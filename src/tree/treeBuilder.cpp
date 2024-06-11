#include "treeBuilder.h"

std::shared_ptr<PublicBeliefState> TreeBuilder::buildDefaultTree(){
    std::shared_ptr<PublicBeliefState> root = std::make_shared<PublicBeliefState>();
    root->setPlayer(Player::P1);
    root->setBets({settings::ante, settings::ante});
    root->setActionToThisNode(ActionType::START);
    root->setTerminal(false);
    root->setDepth(0);
    root->setStreet(1);
    //root->setBoard({Card("As")});

    return buildTreeDFS(root, 100000000);
}

std::shared_ptr<PublicBeliefState> TreeBuilder::buildTree(std::shared_ptr<PublicBeliefState> nodeToBuildFrom, int maxDepth){
    std::shared_ptr<PublicBeliefState> root = std::make_shared<PublicBeliefState>();
    root->setPlayer(nodeToBuildFrom->getPlayer());
    root->setBets(nodeToBuildFrom->getBets());
    root->setActionToThisNode(nodeToBuildFrom->getActionToThisNode());
    root->setTerminal(nodeToBuildFrom->isTerminal());
    root->setDepth(0); // change this to 0 to have dynamic depth evaluation of NN
    root->setStreet(nodeToBuildFrom->getStreet());
    root->setBoard(nodeToBuildFrom->getBoard());

    return buildTreeDFS(root, maxDepth);
}

std::shared_ptr<PublicBeliefState> TreeBuilder::buildDummyTree(std::shared_ptr<PublicBeliefState> chanceNode, Player resolvingPlayer){
    assert(chanceNode->getPlayer() == Player::CHANCE);

    std::shared_ptr<PublicBeliefState> parent = std::make_shared<PublicBeliefState>();
    parent->setPlayer(resolvingPlayer);
    parent->setBets(chanceNode->getBets());
    parent->setActionToThisNode(ActionType::START);
    parent->setTerminal(false);
    parent->setDepth(chanceNode->getDepth());
    parent->setStreet(chanceNode->getStreet());

    std::shared_ptr<PublicBeliefState> chanceClone = std::make_shared<PublicBeliefState>();
    chanceClone->setHasParent(true);
    chanceClone->setPlayer(Player::CHANCE);
    chanceClone->setBets(chanceNode->getBets());
    chanceClone->setActionToThisNode(chanceNode->getActionToThisNode());
    chanceClone->setTerminal(chanceNode->isTerminal());
    chanceClone->setDepth(chanceNode->getDepth());
    chanceClone->setStreet(chanceNode->getStreet());
    chanceClone->setBoard(chanceNode->getBoard());

    parent->setChildren({chanceClone});

    buildTreeDFS(chanceClone, 100000000);

    return parent;
}

std::shared_ptr<PublicBeliefState> TreeBuilder::buildTreeDFS(std::shared_ptr<PublicBeliefState> current, int maxDepth){
    if(maxDepth == 0) return current;

    std::array<int, 2> bets = current->getBets();
    assert(bets[0] > 0);
    assert(bets[1] > 0);

    std::vector<std::shared_ptr<PublicBeliefState>> children = getChildren(current);
    for(auto child : children){
        buildTreeDFS(child, maxDepth - 1);
    }
    current->setChildren(children);
    return current;
}

std::vector<std::shared_ptr<PublicBeliefState>> TreeBuilder::getChildren(std::shared_ptr<PublicBeliefState> current){
    if(current->isTerminal()){
        return {};
    }
    if(current->getPlayer() == Player::CHANCE){
        return getChildrenChance(current);
    }
    return getChildrenPlayer(current);
}

std::vector<std::shared_ptr<PublicBeliefState>> TreeBuilder::getChildrenPlayer(std::shared_ptr<PublicBeliefState> current){
    std::vector<std::shared_ptr<PublicBeliefState>> children;
    Player actingPlayer = current->getPlayer();
    std::array<int, 2> bets = current->getBets();

    // if the bets are the same then the acting player can check
    if(bets[0] == bets[1]){
        std::shared_ptr<PublicBeliefState> check = std::make_shared<PublicBeliefState>();
        check->setHasParent(true);
        check->setPlayer(utils::otherPlayer(actingPlayer));
        check->setBets(bets);
        check->setBoard(current->getBoard());
        if(current->getPlayer() == Player::P1){
            // if the acting player is P1 then betting continues this round
            check->setActionToThisNode(ActionType::CHECK);
        } else {
            // if the acting player is P2 then this round is finished
            check->setActionToThisNode(ActionType::TCHECK);
            check->setPlayer(Player::CHANCE);
            // Temporarily only consider pre-flop
            //check->setTerminal(true);

            // if the current street is the river then the game is over
            if(current->getStreet() == settings::numStreets){
                check->setTerminal(true);
            }
        }

        check->setDepth(current->getDepth() + 1);
        check->setStreet(current->getStreet());
        check->history.push_back("c");

        children.push_back(check);
    } else{
        // if the bets are not the same then the acting player can call or fold
        std::shared_ptr<PublicBeliefState> fold = std::make_shared<PublicBeliefState>();
        fold->setHasParent(true);
        fold->setPlayer(utils::otherPlayer(actingPlayer));
        fold->setBets(bets);
        fold->setActionToThisNode(ActionType::FOLD);
        fold->setTerminal(true);
        fold->setDepth(current->getDepth() + 1);
        fold->setStreet(current->getStreet());
        fold->setBoard(current->getBoard());

        fold->history.push_back("f");

        children.push_back(fold);

        std::shared_ptr<PublicBeliefState> call = std::make_shared<PublicBeliefState>();
        call->setHasParent(true);
        int maxBet = std::max(bets[0], bets[1]);
        std::array<int, 2> newBets = {maxBet, maxBet};
        call->setBets(newBets);
        call->setActionToThisNode(ActionType::CALL);
        call->setDepth(current->getDepth() + 1);
        call->setBoard(current->getBoard());

        call->setStreet(current->getStreet());
        call->setPlayer(Player::CHANCE);

        // if the current street is the river then the game is over
        if(current->getStreet() == settings::numStreets || maxBet == settings::stackSize){
            call->setTerminal(true);
        }

        call->history.push_back("C");

        children.push_back(call);        
    }

    // as long as the max bet is not the stack then the acting player can bet
    // (could call otherwise)
    int maxBet = std::max(bets[0], bets[1]);
    if(maxBet < settings::stackSize){
        std::vector<std::array<int, 2>> possibleBets = getPossibleBets(current);
        for(auto & bet : possibleBets){
            std::shared_ptr<PublicBeliefState> newBet = std::make_shared<PublicBeliefState>();
            newBet->setHasParent(true);
            newBet->setPlayer(utils::otherPlayer(actingPlayer));
            newBet->setBets(bet);
            newBet->setActionToThisNode(ActionType::BET);
            newBet->setDepth(current->getDepth() + 1);
            newBet->setStreet(current->getStreet());
            newBet->setBoard(current->getBoard());

            newBet->history.push_back("b" + std::to_string(bet[static_cast<int>(actingPlayer)]));

            children.push_back(newBet);
        }
    }

    return children;
}


std::vector<std::shared_ptr<PublicBeliefState>> TreeBuilder::getChildrenChance(std::shared_ptr<PublicBeliefState> current){
    const std::vector<Card> currentBoard = current->getBoard();

    std::vector<std::shared_ptr<PublicBeliefState>> children;
    for(const std::string & card : settings::deckCards){
        if(std::find(currentBoard.begin(), currentBoard.end(), Card(card)) != currentBoard.end()){
            // can't have the same card twice
            continue;
        }
        
        std::shared_ptr<PublicBeliefState> child = std::make_shared<PublicBeliefState>();
        child->setHasParent(true);
        child->setPlayer(Player::P1);
        child->setBets(current->getBets());
        child->setActionToThisNode(ActionType::CHANCE);
        std::vector<Card> newBoard = currentBoard;
        newBoard.push_back(Card(card));
        child->setBoard(newBoard);
        child->setStreet(current->getStreet() + 1);
        child->setDepth(current->getDepth() + 1);

        children.push_back(child);
    }

    return children;
}

std::vector<std::array<int, 2>> TreeBuilder::getPossibleBets(std::shared_ptr<PublicBeliefState> current){
    std::array<int, 2> bets = current->getBets();
    Player actingPlayer = current->getPlayer();
    int effectivePot = std::max(bets[0], bets[1]) * 2; // this is what the pot size is considered to be (as if I call first and then can bet)
    int currentBet = std::min(bets[0], bets[1]);

    assert(bets[static_cast<int>(actingPlayer)] == currentBet); // the betting player should not have more in the pot

    std::vector<std::array<int, 2>> possibleBets;
    bool betAllin = false; // do not double count allin bets
    for(auto betSize : settings::betSizing){
        std::array<int, 2> bet = bets;
        int actingBet = betSize * effectivePot;
        actingBet += effectivePot / 2;
        if(actingBet > settings::stackSize && !betAllin){
            actingBet = settings::stackSize;
            betAllin = true;
        }
        if(actingBet >= settings::ante + std::max(bets[0], bets[1]) && actingBet <= settings::stackSize){
            bet[static_cast<int>(actingPlayer)] = actingBet;
            possibleBets.push_back(bet);
        }
    }

    // if(!betAllin){
    //     std::array<int, 2> bet = bets;
    //     bet[static_cast<int>(actingPlayer)] = settings::stackSize;
    //     possibleBets.push_back(bet);
    // }

    return possibleBets;
}
