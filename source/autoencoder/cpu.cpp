#include "autoencoder.hpp"

#include <iostream>

using namespace Autoencoder;

void CPU::update(float learning_rate, cudaStream_t /*stream*/)
{
	for (const auto &layer : layers)
	{
		layer->update(learning_rate);
	}
}

Tensor CPU::forward_encode(const Tensor &input, cudaStream_t stream)
{
	Tensor x = input;
	for (int i = 0; i <= encode_layer; ++i)
	{
		x = layers[i]->forward_cpu(x);
	}
	return x;
}

Tensor CPU::forward_decode(const Tensor &input, cudaStream_t stream)
{
	Tensor x = input;
	for (int i = encode_layer + 1; i < layers.size(); ++i)
	{
		x = layers[i]->forward_cpu(x);
	}
	return x;
}

Tensor CPU::backward_encode(const Tensor &gradient, cudaStream_t stream)
{
	Tensor grad = gradient;
	for (int i = encode_layer; i >= 0; --i)
	{
		std::cout << "[CPU] backward encode layer " << i << std::endl;
		grad = layers[i]->backward_cpu(grad);
	}
	return grad;
}

Tensor CPU::backward_decode(const Tensor &gradient, cudaStream_t stream)
{
	Tensor grad = gradient;
	for (int i = layers.size() - 1; i > encode_layer; --i)
	{
		std::cout << "[CPU] backward decode layer " << i << std::endl;
		grad = layers[i]->backward_cpu(grad);
	}
	return grad;
}