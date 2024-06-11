#include "utils.h"

namespace utils{
    Player otherPlayer(Player player){
        assert(player != Player::CHANCE);
        if(player == Player::P1){
            return Player::P2;
        }
        return Player::P1;
    }

    std::vector<std::string> split(const std::string & s, char delimiter) {
        std::vector<std::string> parts;
        std::string part;
        std::istringstream stream(s);
        while (std::getline(stream, part, delimiter)) {
            parts.push_back(part);
        }
        return parts;
    }

    torch::Tensor toTensor(const std::vector<std::vector<float>> & vec){
        int dim0 = vec.size();
        int dim1 = vec[0].size();
        std::vector<float> flat_vec;
        for (const auto & row : vec){
            for (const auto & val : row){
                flat_vec.push_back(val);
            }
        }
        torch::Tensor tensor = torch::from_blob(flat_vec.data(), {dim0, dim1}, torch::kFloat32);
        return tensor.clone();
    };

    torch::Tensor toTensor(const std::vector<int> & vec){
        std::vector<float> float_vec;
        for (const auto & val : vec){
            float_vec.push_back(static_cast<float>(val));
        }
        at::Tensor tensor = torch::from_blob(float_vec.data(), {static_cast<int>(vec.size())}, torch::kFloat32);
        at::Tensor ten2 = tensor.to(torch::kInt32);
        return ten2.clone();
    };

    torch::Tensor toTensor(const std::vector<float> & vec){
        std::vector<float> float_vec;
        for (const auto & val : vec){
            float_vec.push_back(val);
        }
        torch::Tensor tensor = torch::from_blob(float_vec.data(), {static_cast<int>(vec.size())}, torch::kFloat32);
        return tensor.clone();
    };

    std::string threadId(){
        std::stringstream ss;
        ss << std::this_thread::get_id();
        return ss.str();
    }

    bool handBlockedByBoard(const std::vector<Card> & hand, const std::vector<Card> & board){
        for(const Card & card : hand){
            if(std::find(board.begin(), board.end(), card) != board.end()){
                return true;
            }
        }
        return false;
    }

    int encodeBoard(const std::vector<Card> & board){
        int bitBoard = 0;
        for(const Card & card : board){
            int index = -1;
            for(int i = 0; i < settings::deckSize; i++){
                if(settings::deckCards[i] == card.toString()){
                    index = i;
                    break;
                }
            }
            assert(index != -1);
            bitBoard |= (1 << index);
        }
        return bitBoard;
    }

};

// Overload the << operator to handle the Player enum class
std::ostream& operator<<(std::ostream& os, const Player& player) {
    switch (player) {
        case Player::P1:
            os << "Player P1";
            break;
        case Player::P2:
            os << "Player P2";
            break;
        case Player::CHANCE:
            os << "Chance";
            break;
        default:
            os << "Unknown Player";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ActionType& action){
    switch (action) {
        case ActionType::FOLD:
            os << "FOLD";
            break;
        case ActionType::CHECK:
        case ActionType::TCHECK:
            os << "CHECK";
            break;
        case ActionType::CALL:
            os << "CALL";
            break;
        case ActionType::BET:
            os << "BET";
            break;
        case ActionType::START:
            os << "START";
            break;
        case ActionType::CHANCE:
            os << "CHANCE";
            break;
        default:
            os << "Unknown Action";
            break;
    }
    return os;
}
