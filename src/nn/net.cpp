#include "net.h"

NetImpl::NetImpl(int input_size_, int hidden_size_, int output_size_){
    input_size = input_size_;
    hidden_size = hidden_size_;
    output_size = output_size_;

    fc1 = torch::nn::Linear(input_size, hidden_size);
    fc2 = torch::nn::Linear(hidden_size, hidden_size);
    fc3 = torch::nn::Linear(hidden_size, hidden_size);
    fc4 = torch::nn::Linear(hidden_size, output_size);

    // Initialize BatchNorm layers
    // bn1 = torch::nn::BatchNorm1d(hidden_size);
    // bn2 = torch::nn::BatchNorm1d(hidden_size);
    // bn3 = torch::nn::BatchNorm1d(hidden_size);
    // drop2 = torch::nn::Dropout(0.05);
    // drop2 = torch::nn::Dropout(0.05);
    // drop3 = torch::nn::Dropout(0.05);

    ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));
    ln3 = torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}));

    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    // register_module("bn1", bn1);
    // register_module("bn2", bn2);
    // register_module("bn3", bn3);
    // register_module("drop2", drop2);
    // register_module("drop3", drop3);

    register_module("ln1", ln1);
    register_module("ln2", ln2);
    register_module("ln3", ln3);
}

torch::Tensor NetImpl::forward(torch::Tensor x){
    // x has shape [batch_size, input_size]
    // x = torch::relu(bn1(fc1->forward(x)));
    // x = torch::relu(bn2(fc2->forward(x)));
    // x = torch::relu(bn3(fc3->forward(x)));
    // get the ranges
    // torch::Tensor range1 = x.index({torch::indexing::Slice(), torch::indexing::Slice(11, 26)}).clone();
    // torch::Tensor range2 = x.index({torch::indexing::Slice(), torch::indexing::Slice(26, 41)}).clone();
    // torch::Tensor board = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 6)}).clone();
    // auto now2 = std::chrono::high_resolution_clock::now();

    torch::Tensor range1 = x.index({torch::indexing::Slice(), torch::indexing::Slice(settings::deckSize + 5, settings::deckSize + 5 + settings::numPossibleHands)}).clone();
    torch::Tensor range2 = x.index({torch::indexing::Slice(), torch::indexing::Slice(settings::deckSize + 5 + settings::numPossibleHands, settings::deckSize + 5 + settings::numPossibleHands * 2)}).clone();
    torch::Tensor board = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, settings::deckSize)}).clone();

    // auto start = std::chrono::high_resolution_clock::now();
    //x = torch::prelu(fc1->forward(x), torch::full({1}, 0.25));
    x = netHelper::customGELU(fc1->forward(x));
    x = ln1(x);
    x = netHelper::customGELU(fc2->forward(x));
    //x = torch::prelu(fc2->forward(x), torch::full({1}, 0.25));
    // x = drop2(x);
    x = ln2(x);
    x = netHelper::customGELU(fc3->forward(x));
    //x = torch::prelu(fc3->forward(x), torch::full({1}, 0.25));
    x = ln3(x);
    // x = drop2(x);
    x = fc4->forward(x);
    // auto stop = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - now2);
    // std::cout << "Time taken by function: " << duration.count() << " nanos" << std::endl;

    // mask out impossible board
    // x = x.view({x.size(0), 2, -1});
    // // torch::Tensor rangeMaskBoard = netHelper::getBoardMask(board[0]);
    // // x = x * rangeMaskBoard.view({x.size(0), 1, -1});
    // x = x * (1 - board.view({x.size(0), 1, -1}));
    // x = x.view({x.size(0), -1});

    torch::Tensor y = x.view({x.size(0), 2, -1}).clone();

    y.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}) *= range1;
    y.index({torch::indexing::Slice(), 1, torch::indexing::Slice()}) *= range2;

    y = torch::sum(y, 2);
    y = torch::sum(y, 1);

    y = y.view({-1, 1});
    x -= (y/2);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - now);
    // ppr("Inference time:", duration.count());

    // x = x.view({x.size(0), 2, -1});
    // x = x * (1 - board.view({x.size(0), 1, -1}));
    // // x = x * rangeMaskBoard.view({x.size(0), 1, -1});
    // x = x.view({x.size(0), -1});
    
    // torch::Tensor y2 = x.view({x.size(0), 2, -1}).clone();
    // y2.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}) *= range1;
    // y2.index({torch::indexing::Slice(), 1, torch::indexing::Slice()}) *= range2;
    // y2 = torch::sum(y2, 2);

    // std::cout << y2 << std::endl;

    // y2 = torch::sum(y2, 1);

    // std::cout << "AFT: " << y2 << std::endl;

    // assert(false);

    return x;
}