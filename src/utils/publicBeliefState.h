#pragma once

#include "utils.h"
#include "card.h"

#include <memory>
#include <vector>
#include <array>

#include <torch/torch.h>

class PublicBeliefState {

public:
    PublicBeliefState() : hasParent(false), terminal(false), depth(0), street(0), bets({0, 0}) {}

    bool hasParentNode() const { return hasParent; }
    std::vector<std::shared_ptr<PublicBeliefState>> getChildren() const { return children; }
    bool isTerminal() const { return terminal; }
    int getDepth() const { return depth; }
    std::vector<Card> getBoard() const { return board; }
    Player getPlayer() const { return player; }
    std::array<int, 2> getBets() const { return bets; }
    ActionType getActionToThisNode() const { return actionToThisNode; }
    int getStreet() const { return street; }
    int getPotSize() const { return std::min(bets[0], bets[1]); }
    void addChild(std::shared_ptr<PublicBeliefState> child) { children.push_back(child); }
    void setChildren(std::vector<std::shared_ptr<PublicBeliefState>> children_) { this->children = children_; }
    void setHasParent(bool hasParent_) { this->hasParent = hasParent_; }
    void setTerminal(bool terminal_) { this->terminal = terminal_; }
    void setDepth(int depth_) { this->depth = depth_; }
    void setBoard(std::vector<Card> board_) { this->board = board_; }
    void setPlayer(Player player_) { this->player = player_; }
    void setBets(std::array<int, 2> bets_) { this->bets = bets_; }
    void setActionToThisNode(ActionType actionToThisNode_) { this->actionToThisNode = actionToThisNode_; }
    void setStreet(int street_) { this->street = street_; }
    void printImp() const;

    std::pair<std::string, std::string> nodeToGraphViz();

    torch::Tensor strategy; // Strategy for the acting player, CxA where C is the number of infosets (private cards) and A is the number of valid actions
    torch::Tensor cumulativeStrategy;
    torch::Tensor ranges; // 2xC where C is the number of infosets (private cards)

    torch::Tensor cfValues; // values in self-play (Player x Infoset)
    torch::Tensor cumulativeCFValues;
    torch::Tensor cfValuesBestReponse; // values vs best response
    std::vector<std::string> history;
    std::vector<torch::Tensor> strategiesAtCFR;

    torch::Tensor regrets;
    torch::Tensor iter_weight_sum;

    float exploitability;
    float epsilonP1;
    float epsilonP2;
private:
    bool hasParent;
    std::vector<std::shared_ptr<PublicBeliefState>> children;
    Player player;
    ActionType actionToThisNode;
    bool terminal;
    int depth;
    int street;
    std::vector<Card> board;
    std::array<int, 2> bets;


};

std::ostream& operator<<(std::ostream& os, const PublicBeliefState& pbs);
