#include "autoencoder.hpp"

#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <iostream>

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

void Base::save_model(const std::string &file_path) {
	std::ofstream out(file_path, std::ios::binary);
	if (!out) {
		throw std::runtime_error("Failed to open file for saving model: " + file_path);
	}

	// Simple binary format: magic, layer count, then per-layer weight/bias sizes and data
	const std::uint32_t magic = 0x41455631; // "AEV1"
	const std::uint32_t layer_count = static_cast<std::uint32_t>(layers.size());
	out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
	out.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

	for (const auto &layer_ptr : layers) {
		Layer* layer = layer_ptr.get();

		// Preserve device residency: load/save on CPU, then restore if needed
		const bool weights_were_gpu = layer->weights.is_gpu();
		const bool biases_were_gpu = layer->biases.is_gpu();
		if (weights_were_gpu) layer->weights.to_cpu();
		if (biases_were_gpu) layer->biases.to_cpu();

		const std::size_t w_size = layer->weights.size();
		const std::size_t b_size = layer->biases.size();

		out.write(reinterpret_cast<const char*>(&w_size), sizeof(w_size));
		out.write(reinterpret_cast<const char*>(&b_size), sizeof(b_size));

		if (w_size > 0) {
			out.write(reinterpret_cast<const char*>(layer->weights.data()), sizeof(float) * w_size);
		}
		if (b_size > 0) {
			out.write(reinterpret_cast<const char*>(layer->biases.data()), sizeof(float) * b_size);
		}

		if (weights_were_gpu) layer->weights.to_gpu();
		if (biases_were_gpu) layer->biases.to_gpu();
	}

	std::cout << "[Autoencoder] Model saved to " << file_path << std::endl;
}

void Base::load_model(const std::string &file_path) {
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("Failed to open file for loading model: " + file_path);
	}

	std::uint32_t magic = 0;
	std::uint32_t layer_count = 0;
	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	in.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));

	if (magic != 0x41455631) {
		throw std::runtime_error("Invalid model file (bad magic)");
	}
	if (layer_count != layers.size()) {
		throw std::runtime_error("Layer count mismatch when loading model");
	}

	for (std::size_t idx = 0; idx < layers.size(); ++idx) {
		Layer* layer = layers[idx].get();

		const bool weights_were_gpu = layer->weights.is_gpu();
		const bool biases_were_gpu = layer->biases.is_gpu();
		if (weights_were_gpu) layer->weights.to_cpu();
		if (biases_were_gpu) layer->biases.to_cpu();

		std::size_t w_size_file = 0;
		std::size_t b_size_file = 0;
		in.read(reinterpret_cast<char*>(&w_size_file), sizeof(w_size_file));
		in.read(reinterpret_cast<char*>(&b_size_file), sizeof(b_size_file));

		if (w_size_file != layer->weights.size() || b_size_file != layer->biases.size()) {
			throw std::runtime_error("Model parameter size mismatch when loading layer " + std::to_string(idx));
		}

		if (w_size_file > 0) {
			in.read(reinterpret_cast<char*>(layer->weights.data()), sizeof(float) * w_size_file);
		}
		if (b_size_file > 0) {
			in.read(reinterpret_cast<char*>(layer->biases.data()), sizeof(float) * b_size_file);
		}

		if (!in) {
			throw std::runtime_error("Unexpected end of file while loading model");
		}

		if (weights_were_gpu) layer->weights.to_gpu();
		if (biases_were_gpu) layer->biases.to_gpu();
	}

	std::cout << "[Autoencoder] Model loaded from " << file_path << std::endl;
}