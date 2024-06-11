#include "trainer.h"

Trainer::Trainer(std::vector<std::string> playerFiles_, bool loadModel_, int num_epochs_){
    playerFiles = playerFiles_;
    loadModel = loadModel_;
    num_epochs = num_epochs_;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<std::pair<torch::Tensor, torch::Tensor>>> Trainer::read_training_data(){
    std::vector<torch::Tensor> allTargets;
    std::vector<torch::Tensor> features;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> validation;

    for(std::string filename : playerFiles){
        std::ifstream stream(filename);
        if(!stream.is_open()){
            std::cerr << "Could not open file: " << filename << std::endl;
            exit(1);
        }

        std::string line;
        std::vector<torch::Tensor> featureRows;
        while(std::getline(stream, line)){
            std::vector<std::string> parts = utils::split(line, '|');
            std::vector<float> featureRow;
            for(size_t i = 0; i < parts.size(); i++){
                double val = std::stod(parts[i]);
                float floatVal = static_cast<float>(val);
                featureRow.push_back(floatVal);
            }
            featureRows.push_back(utils::toTensor(featureRow));
        }
        stream.close();
        torch::Tensor targets;
        torch::load(targets, filename + ".targets");
        allTargets.push_back(targets);
        features.push_back(torch::stack(featureRows, 0));
    }

    assert(allTargets.size() == features.size());
    std::vector<torch::Tensor> finalFeatures;
    std::vector<torch::Tensor> finalTargets;

    for(size_t i = 0; i < allTargets.size(); i++){
        assert(allTargets[i].size(0) == features[i].size(0));
        // get 90% of the data
        int train_size = features[i].size(0) * 0.9;
        torch::Tensor train_x = features[i].slice(0, 0, train_size);
        torch::Tensor train_y = allTargets[i].slice(0, 0, train_size);
        torch::Tensor val_x = features[i].slice(0, train_size, features[i].size(0));
        torch::Tensor val_y = allTargets[i].slice(0, train_size, allTargets[i].size(0));

        validation.push_back({val_x, val_y});
        finalFeatures.push_back(train_x);
        finalTargets.push_back(train_y);
    }

    torch::Tensor allFeatures = torch::cat(finalFeatures, 0);
    torch::Tensor targets = torch::cat(finalTargets, 0);

    return {allFeatures, targets, validation};
}

void Trainer::getLoss(std::shared_ptr<Net> model, const std::vector<std::pair<torch::Tensor, torch::Tensor>> & validation){
    (*model)->eval();

    for(size_t i = 0; i < validation.size(); i++){
        auto criterion = torch::nn::HuberLoss();

        torch::Tensor val_x = validation[i].first;
        torch::Tensor val_y = validation[i].second;

        if(settings::useCuda && torch::cuda::is_available()){
            val_x = val_x.to(torch::kCUDA);
            val_y = val_y.to(torch::kCUDA);
        } else if(settings::useMPS && torch::mps::is_available()){
            val_x = val_x.to(torch::kMPS);
            val_y = val_y.to(torch::kMPS);
        }

        auto loss = criterion((*model)->forward(val_x), val_y);
        std::string msg = "Loss on the validation set at file " + std::to_string(i+1) + " is: " + std::to_string(loss.to(torch::kCPU).item<float>());
        pr(msg);
    }
}

void Trainer::train(std::shared_ptr<Net> model, float lr, int currentName){
    auto [features, targets, validation] = read_training_data();
    std::cout << features.sizes() << std::endl;
    std::cout << targets.sizes() << std::endl;

    // move data to GPU
    if(settings::useCuda && torch::cuda::is_available()){
        pr("CUDA is available!");
        features = features.to(torch::kCUDA);
        targets = targets.to(torch::kCUDA);
    } else if(settings::useMPS && torch::mps::is_available()){
        pr("MPS is available!");
        features = features.to(torch::kMPS);
        targets = targets.to(torch::kMPS);
    } else {
        pr("Not using CUDA or MPS")
    }

    // move model to cuda
    if(settings::useCuda && torch::cuda::is_available()){
        pr("Moving model to CUDA");
        (*model)->to(torch::kCUDA);
    } else if(settings::useMPS && torch::mps::is_available()){
        pr("Moving model to MPS");
        (*model)->to(torch::kMPS);
    }

    // shuffle data
    auto indices = torch::randperm(features.size(0));
    // move indicies to GPU
    if(settings::useCuda && torch::cuda::is_available()){
        indices = indices.to(torch::kCUDA);
    } else if(settings::useMPS && torch::mps::is_available()){
        indices = indices.to(torch::kMPS);
    }

    auto train_x = features.index_select(0, indices);
    auto train_y = targets.index_select(0, indices);

    torch::optim::Adam optimizer((*model)->parameters(), torch::optim::AdamOptions(lr).weight_decay(0.0001));

    if(loadModel){
        torch::load(optimizer, "optimizer.pt");
    }

    //auto criterion = torch::nn::MSELoss();
    auto criterion = torch::nn::HuberLoss();

    getLoss(model, validation);

    int batch_size = 100;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {


        // shuffle data
        // auto indTrain = torch::randperm(train_x.size(0));
        // // move indicies to GPU
        // if(torch::cuda::is_available()){
        //     indTrain = indTrain.to(torch::kCUDA);
        // }
        // train_x = train_x.index_select(0, indTrain);
        // train_y = train_y.index_select(0, indTrain);

        (*model)->train();
        int64_t num_batches = train_x.size(0) / batch_size;
        torch::Tensor epoch_loss = torch::zeros({1});
        if(settings::useCuda && torch::cuda::is_available()){
            epoch_loss = epoch_loss.to(torch::kCUDA);
        } else if(settings::useMPS && torch::mps::is_available()){
            epoch_loss = epoch_loss.to(torch::kMPS);
        }

        for (int64_t batch = 0; batch < num_batches; ++batch) {
            auto start = batch * batch_size;
            auto end = std::min(start + batch_size, train_x.size(0));
            auto x_batch = train_x.slice(0, start, end);
            auto y_batch = train_y.slice(0, start, end);

            auto predictions = (*model)->forward(x_batch);

            auto loss = criterion(predictions, y_batch);
            epoch_loss += loss;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        if (epoch % 1 == 0) {
            std::string msg = "Epoch [" + std::to_string(epoch + 1) + "/" + 
                                            std::to_string(num_epochs) + "], Loss: " + 
                                            std::to_string(epoch_loss.to(torch::kCPU).item<float>() / num_batches);
            pr(msg);
        }

        if(epoch == num_epochs - 1){
            getLoss(model, validation);
            torch::save(*model, settings::modelName);
            torch::save(*model, "models/modelIter" + std::to_string(currentName) + ".pt");
            torch::save(optimizer, "optimizer.pt");
        }
    }

}
