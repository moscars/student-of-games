#include "dataSaver.h"

namespace dataSaver {
    void saveData(torch::Tensor data, bool doNext){
        std::lock_guard<std::mutex> lock(settings::saveMutex);
        std::ofstream file;//("data/" + settings::fileName, std::ios::app);
        if(doNext){
            std::vector<std::string> parts = utils::split(settings::fileName, ':');

            std::string tempFileName = parts[0];
            tempFileName += ":" + parts[1];

            std::string end = parts[2];
            assert(parts.size() == 3);

            std::vector<std::string> endParts = utils::split(end, '.');

            tempFileName += ":" + std::to_string(std::stoi(endParts[0]) + 1);
            tempFileName += "." + endParts[1];

            file.open("data/" + tempFileName, std::ios::app);
        } else {
            file.open("data/" + settings::fileName, std::ios::app);
        }

        if(!file.is_open()){
            throw std::runtime_error("Could not open file to save data");
        }

        for(int i = 0; i < data.size(0); i++){
            file << data[i].item<float>() << "|";
        }
        file << std::endl;
        file.close();
    }
};