#include "layer.hpp"

// Tensor ReLU::forward(const Tensor& input)
// {
//     Tensor output;
//     // output.initialize(input.getSize());
    
//     int size = input.size();
//     for (int i = 0; i < size; ++i)
//     {
//         float value = input[i];
//         output[i] = value > 0 ? value : 0; // ReLU activation: max(0, x)
//     }

//     return output;
// }