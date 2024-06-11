#pragma once

#include "card.h"
#include "../settings.h"

#include <iostream>
#include <ostream>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>

#include <torch/torch.h>

#define pr(x) { \
    std::lock_guard<std::mutex> lock(settings::printMutex); \
    std::ofstream log_file("log.txt", std::ios::app); \
    auto now = std::chrono::system_clock::now(); \
    time_t now_t = std::chrono::system_clock::to_time_t(now); \
    std::tm * local_time = std::localtime(&now_t); \
    char buffer[80]; \
    std::strftime(buffer, sizeof(buffer), "%b %d %H:%M:%S", local_time); \
    std::cout << buffer << " " << x << std::endl; \
    log_file << buffer << " " << x << std::endl; \
    log_file.close(); \
};

#define ppr(x,y) { \
    std::lock_guard<std::mutex> lock(settings::printMutex); \
    std::ofstream log_file("log.txt", std::ios::app); \
    auto now = std::chrono::system_clock::now(); \
    time_t now_t = std::chrono::system_clock::to_time_t(now); \
    std::tm* local_time = std::localtime(&now_t); \
    char buffer[80]; \
    std::strftime(buffer, sizeof(buffer), "%b %d %H:%M:%S", local_time); \
    std::cout << buffer << " " << x << " " << y << std::endl; \
    log_file << buffer << " " << x << " " << y << std::endl; \
    log_file.close(); \
};

enum class Player{
    P1 = 0,
    P2 = 1,
    CHANCE = 2,
};

std::ostream& operator<<(std::ostream& os, const Player& player);

enum class ActionType{
    FOLD = 0,
    CHECK = 1,
    TCHECK = 2,
    CALL = 3,
    BET = 4,
    START = 5,
    CHANCE = 6,
};

std::ostream& operator<<(std::ostream& os, const ActionType& action);

namespace utils{
    Player otherPlayer(Player player);
    bool handBlockedByBoard(const std::vector<Card> & hand, const std::vector<Card> & board);
    std::vector<std::string> split(const std::string & s, char delimiter);
    torch::Tensor toTensor(const std::vector<std::vector<float>> & vec);
    torch::Tensor toTensor(const std::vector<int> & vec);
    torch::Tensor toTensor(const std::vector<float> & vec);
    std::string threadId();
    int encodeBoard(const std::vector<Card> & board);
};