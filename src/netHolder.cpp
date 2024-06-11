#include "netHolder.h"

namespace netHolder{
    std::shared_ptr<Net> model = nullptr;
    std::shared_ptr<Net> modelP1 = nullptr;
    std::shared_ptr<Net> modelP2 = nullptr;
    std::shared_ptr<EquityCache> equityCache = nullptr;
};