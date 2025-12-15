#include "autoencoder.hpp"

using namespace Autoencoder;

void CPU::update(float learning_rate) {
	for (const auto& layer : layers) {
		layer->update(learning_rate);
	}
}

Tensor CPU::forward_encode(const Tensor &input) {
	Tensor x = input;
	for (int i = 0; i <= encode_layer; ++i) {
		x = layers[i]->forward_cpu(x);
	}
	return x;
}

Tensor CPU::forward_decode(const Tensor &input) {
	Tensor x = input;
	for (int i = encode_layer + 1; i < layers.size(); ++i) {
		x = layers[i]->forward_cpu(x);
	}
	return x;
}

Tensor CPU::backward_encode(const Tensor &gradient) {
	Tensor grad = gradient;
	for (int i = encode_layer; i >= 0; --i) {
		grad = layers[i]->backward_cpu(grad);
	}
	return grad;
}

Tensor CPU::backward_decode(const Tensor &gradient) {
	Tensor grad = gradient;
	for (int i = layers.size() - 1; i > encode_layer; --i) {
		grad = layers[i]->backward_cpu(grad);
	}
	return grad;
}