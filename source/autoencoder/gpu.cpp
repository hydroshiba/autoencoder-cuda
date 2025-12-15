#include "autoencoder.hpp"
#include <iostream>

using namespace Autoencoder;

GPU::GPU() : Base()
{
	for (auto &layer : layers)
	{
		layer->to_gpu();
	}

	std::cout << "[Autoencoder::GPU] Model moved to GPU." << std::endl;
}

void GPU::update(float learning_rate)
{
	for (const auto &layer : layers)
	{
		layer->update(learning_rate);
	}
}

Tensor GPU::forward_encode(const Tensor &input)
{
	Tensor x = Tensor(input).to_gpu();
	for (int i = 0; i <= encode_layer; ++i)
	{
		x = layers[i]->forward_gpu(x);
	}
	return x;
}
Tensor GPU::forward_decode(const Tensor &input)
{
	Tensor x = Tensor(input).to_gpu();
	for (int i = encode_layer + 1; i < layers.size(); ++i)
	{
		x = layers[i]->forward_gpu(x);
	}
	return x;
}

Tensor GPU::backward_encode(const Tensor &gradient)
{
	Tensor grad = Tensor(gradient).to_gpu();
	for (int i = encode_layer; i >= 0; --i)
	{
		grad = layers[i]->backward_gpu(grad);
	}
	return grad;
}

Tensor GPU::backward_decode(const Tensor &gradient)
{
	Tensor grad = Tensor(gradient).to_gpu();
	for (int i = layers.size() - 1; i > encode_layer; --i)
	{
		grad = layers[i]->backward_gpu(grad);
	}
	return grad;
}

// GPU::GPU(const Base &base) : Base(base) {
// 	for(int i = 0; i < layers.size(); i++) {
// 		layers[i].get() = base.layers[i].get();
// 		layers[i]->to_gpu();
// 	}
// }