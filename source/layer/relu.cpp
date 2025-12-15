#include "layer.hpp"

#include <algorithm>

Tensor ReLU::forward_cpu(const Tensor &input) {
	cached_input = input;
	Tensor output(input);
	output = output.to_cpu();

	float* output_data = output.data();
	int size = output.size();
	for (int i = 0; i < size; ++i)
	{
		output_data[i] = std::max(0.0f, output_data[i]);
	}

	return output;
}

Tensor ReLU::backward_cpu(const Tensor &grad_output) {
	Tensor grad_input(grad_output);
	grad_input = grad_input.to_cpu();

	// Gate gradient: pass gradient only where forward input > 0
	int size = grad_input.size();
	float* cached_input_data = cached_input.data();
	float* grad_input_data = grad_input.data();
	for (int i = 0; i < size; ++i) {
		if (cached_input_data[i] <= 0.0f) {
			grad_input_data[i] = 0.0f;
		}
	}

	return grad_input;
}