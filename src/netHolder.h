#pragma once

#include "nn/net.h"
#include "game/equityCache.h"

#include <memory>

namespace netHolder{
    extern std::shared_ptr<Net> model;
    extern std::shared_ptr<Net> modelP1;
    extern std::shared_ptr<Net> modelP2;
    extern std::shared_ptr<EquityCache> equityCache;
};