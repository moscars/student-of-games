#pragma once

#include "../settings.h"
#include "../utils/utils.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <sstream>
#include <string>

namespace dataSaver {
    void saveData(torch::Tensor data, bool doNext);
};