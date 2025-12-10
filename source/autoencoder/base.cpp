#include "autoencoder.hpp"

using namespace Autoencoder;

Base::Base() : layers() {
	this->build();
}

void Base::build() {
	layers.clear();

	// Encode layers
	layers.push_back(std::make_unique<Conv2D>(3, 256, 3, 1, 1));
	layers.push_back(std::make_unique<ReLU>());
	layers.push_back(std::make_unique<MaxPool2D>(2, 2));

	layers.push_back(std::make_unique<Conv2D>(256, 128, 3, 1, 1));
	layers.push_back(std::make_unique<ReLU>());
	layers.push_back(std::make_unique<MaxPool2D>(2, 2));

	encode_layer = layers.size() - 1;

	// Decode layers
	layers.push_back(std::make_unique<Conv2D>(128, 128, 3, 1, 1));
	layers.push_back(std::make_unique<ReLU>());
	layers.push_back(std::make_unique<UpSample2D>(2));

	layers.push_back(std::make_unique<Conv2D>(128, 256, 3, 1, 1));
	layers.push_back(std::make_unique<ReLU>());
	layers.push_back(std::make_unique<UpSample2D>(2));

	layers.push_back(std::make_unique<Conv2D>(256, 3, 3, 1, 1));
}

Tensor Base::forward(const Tensor &input) {
	return forward_decode(forward_encode(input));
}

Tensor Base::encode(const Tensor &input) {
	return forward_encode(input);
}

void Base::backward(const Tensor &gradient) {
	backward_encode(backward_decode(gradient));
}