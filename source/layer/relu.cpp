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

// Placeholder implementations

Tensor ReLU::forward_cpu(const Tensor &input) {
	Tensor output;
	return output;
}

// Tensor ReLU::backward_cpu(const Tensor &grad_output) {
// 	Tensor grad_input;
// 	return grad_input;
// }