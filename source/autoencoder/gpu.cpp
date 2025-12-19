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
	std::cout << "[Autoencoder::GPU] All data stays on GPU - zero CPU-GPU transfers during training!" << std::endl;
}

void GPU::update(float learning_rate)
{
	std::cout << "[GPU::update] Starting update for " << layers.size() << " layers" << std::endl;
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::cout << "[GPU::update] Updating layer " << i << std::endl;
		layers[i]->update(learning_rate);
		std::cout << "[GPU::update] Layer " << i << " updated" << std::endl;
	}
	std::cout << "[GPU::update] All layers updated" << std::endl;
}

Tensor GPU::forward_encode(const Tensor &input, cudaStream_t stream)
{
	// Use standard loop execution (graphs capture full forward pass)
	Tensor x = Tensor(input).to_gpu();
	for (int i = 0; i <= encode_layer; ++i)
	{
		x = layers[i]->forward_gpu(x, stream);
	}
	return x;
}
Tensor GPU::forward_decode(const Tensor &input, cudaStream_t stream)
{
	// Use standard loop execution (graphs capture full forward pass)
	Tensor x = Tensor(input).to_gpu();
	for (int i = encode_layer + 1; i < layers.size(); ++i)
	{
		x = layers[i]->forward_gpu(x, stream);
	}
	return x;
}

Tensor GPU::backward_encode(const Tensor &gradient, cudaStream_t stream)
{
	Tensor grad = Tensor(gradient).to_gpu();
	for (int i = encode_layer; i >= 0; --i)
	{
		grad = layers[i]->backward_gpu(grad, stream);
	}
	return grad;
}

Tensor GPU::backward_decode(const Tensor &gradient, cudaStream_t stream)
{
	Tensor grad = Tensor(gradient).to_gpu();
	for (int i = layers.size() - 1; i > encode_layer; --i)
	{
		grad = layers[i]->backward_gpu(grad, stream);
	}
	return grad;
}

// GPU::GPU(const Base &base) : Base(base) {
// 	for(int i = 0; i < layers.size(); i++) {
// 		layers[i].get() = base.layers[i].get();
// 		layers[i]->to_gpu();
// 	}
// }