#pragma once

#include "../utils/publicBeliefState.h"
#include "../game/gameState.h"
#include "../settings.h"

#include <memory>

#include <torch/torch.h>

namespace treeutils{
    void fillUniformStrategy(std::shared_ptr<PublicBeliefState> node, int maxDepth = 100000000);
    std::shared_ptr<PublicBeliefState> copyNodeFromState(GameState state);
    GameState copyStateFromNode(std::shared_ptr<PublicBeliefState> node);
    template <typename T>
    int calculateTreeSize(std::shared_ptr<T> node){
        int size = 1;
        for(auto child : node->getChildren()){
            size += calculateTreeSize(child);
        }
        return size;
    }
};