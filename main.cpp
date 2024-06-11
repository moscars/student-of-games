#include "src/utils/publicBeliefState.h"
#include "src/utils/card.h"
#include "src/utils/utils.h"
#include "src/tree/treeBuilder.h"
#include "src/tree/treeutils.h"
#include "src/tree/exploitability.h"
#include "src/utils/visualizeTree.h"
#include "src/tree/resolvingStrategyFilling.h"
#include "src/solving/pbscfr.h"
#include "src/nn/net.h"
#include "src/nn/situationSolver.h"
#include "src/nn/dataGenerator.h"
#include "src/nn/trainer.h"
#include "src/settings.h"
#include "src/netHolder.h"
#include "src/solving/deepResolving.h"
#include "src/game/dealer.h"
#include "src/settings.h"

#include <iostream>
#include <torch/torch.h>

int main(){    
    netHolder::equityCache = std::make_shared<EquityCache>();
    Net model = Net(settings::networkInputSize, settings::hiddenSize, settings::networkOutputSize);
    netHolder::model = std::make_shared<Net>(model);

    Dealer dealer;
    dealer.playHand();

    return 0;
}