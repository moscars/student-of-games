#pragma once

#include "../utils/publicBeliefState.h"
#include "../utils/utils.h"
#include "../settings.h"

#include <memory>

class TreeBuilder {

public:
    TreeBuilder() = default;
    std::shared_ptr<PublicBeliefState> buildDefaultTree();
    std::shared_ptr<PublicBeliefState> buildTree(std::shared_ptr<PublicBeliefState> nodeToBuildFrom, int maxDepth = 100000000);
    std::shared_ptr<PublicBeliefState> buildDummyTree(std::shared_ptr<PublicBeliefState> chanceNode, Player resolvingPlayer);


private:
    std::shared_ptr<PublicBeliefState> buildTreeDFS(std::shared_ptr<PublicBeliefState> current, int maxDepth);
    std::vector<std::shared_ptr<PublicBeliefState>> getChildren(std::shared_ptr<PublicBeliefState> current);
    std::vector<std::shared_ptr<PublicBeliefState>> getChildrenChance(std::shared_ptr<PublicBeliefState> current);
    std::vector<std::shared_ptr<PublicBeliefState>> getChildrenPlayer(std::shared_ptr<PublicBeliefState> current);
    std::vector<std::array<int, 2>> getPossibleBets(std::shared_ptr<PublicBeliefState> current);

};