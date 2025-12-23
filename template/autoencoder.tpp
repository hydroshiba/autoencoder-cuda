#include <fstream>
#include <stdexcept>

#include "utils/logger.hpp"
#include "activation.cuh"

template <typename Tag>
Autoencoder<Tag>::Autoencoder() {
	this->build();
	LOG("Model built with", layers.size(), "layers.");
}

template <typename Tag>
Autoencoder<Tag>::Autoencoder(const Autoencoder<Tag> &other) {
	layers.clear();
	layers.reserve(other.layers.size());
	for(const auto &layer : other.layers) layers.push_back(layer->clone());
	encode_layer = other.encode_layer;
}

template <typename Tag>
Autoencoder<Tag>& Autoencoder<Tag>::operator=(const Autoencoder<Tag> &other) {
	if(this == reinterpret_cast<const Autoencoder<Tag>*>(&other)) return *this;
	layers.clear();
	layers.reserve(other.layers.size());
	for(const auto &layer : other.layers) layers.push_back(layer->clone());
	encode_layer = other.encode_layer;
	return *this;
}

template <typename Tag>
void Autoencoder<Tag>::build() {
	layers.clear();

	// Encode layers
	layers.push_back(std::make_unique<Layer::Conv2D<Tag>>(3, 256, 3, 1, 1, Activation::ReLU()));
	layers.push_back(std::make_unique<Layer::MaxPool2D<Tag>>(2, 2));

	layers.push_back(std::make_unique<Layer::Conv2D<Tag>>(256, 128, 3, 1, 1, Activation::ReLU()));
	layers.push_back(std::make_unique<Layer::MaxPool2D<Tag>>(2, 2));

	encode_layer = (int)layers.size() - 1;

	// Decode layers
	layers.push_back(std::make_unique<Layer::Conv2D<Tag>>(128, 128, 3, 1, 1, Activation::ReLU()));
	layers.push_back(std::make_unique<Layer::UpSample2D<Tag>>(2));

	layers.push_back(std::make_unique<Layer::Conv2D<Tag>>(128, 256, 3, 1, 1, Activation::ReLU()));
	layers.push_back(std::make_unique<Layer::UpSample2D<Tag>>(2));

	// Final layer uses Identity (default)
	layers.push_back(std::make_unique<Layer::Conv2D<Tag>>(256, 3, 3, 1, 1));
	layers.back()->parameters()[1]->fill(0.5f); // Set bias to 0.5 to center outputs
}

template <typename Tag>
void Autoencoder<Tag>::clear_gradients() {
	for(auto &layer : layers) {
		layer->clear_gradients();
	}
}

template <typename Tag>
std::vector<std::unique_ptr<Layer::Base<Tag>>>& Autoencoder<Tag>::get_layers() {
	return layers;
}

// Forward / Backward Implementation

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::forward_encode(const Tensor<Tag> &input) {
	Tensor<Tag> x = input;
	for(int i = 0; i <= encode_layer; ++i) {
		x = layers[i]->forward(x);
	}
	return x;
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::forward_decode(const Tensor<Tag> &input) {
	Tensor<Tag> x = input;
	for(size_t i = encode_layer + 1; i < layers.size(); ++i) {
		x = layers[i]->forward(x);
	}
	return x;
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::forward(const Tensor<Tag> &input) {
	return forward_decode(forward_encode(input));
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::encode(const Tensor<Tag> &input) {
	return forward_encode(input);
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::decode(const Tensor<Tag> &input) {
	return forward_decode(input);
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::backward_encode(const Tensor<Tag> &gradient) {
	Tensor<Tag> grad = gradient;
	for(int i = encode_layer; i >= 0; --i) {
		grad = layers[i]->backward(grad);
	}
	return grad;
}

template <typename Tag>
Tensor<Tag> Autoencoder<Tag>::backward_decode(const Tensor<Tag> &gradient) {
	Tensor<Tag> grad = gradient;
	for(int i = (int)layers.size() - 1; i > encode_layer; --i) {
		grad = layers[i]->backward(grad);
	}
	return grad;
}

template <typename Tag>
void Autoencoder<Tag>::backward(const Tensor<Tag> &gradient) {
	backward_encode(backward_decode(gradient));
}

// IO Implementation

template <typename Tag>
void Autoencoder<Tag>::save(const std::string &file_path) {
	std::ofstream out(file_path, std::ios::binary);
	if(!out) throw std::runtime_error("Failed to open file: " + file_path);

	const std::uint32_t magic = 0x41455632; // "AEV2"
	const std::uint32_t layer_count = static_cast<std::uint32_t>(layers.size());
	
	out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
	out.write(reinterpret_cast<const char *>(&layer_count), sizeof(layer_count));

	for(const auto &layer : layers) {
		auto params = layer->parameters();
		
		Tensor<Tag>* w = (params.size() >= 1) ? params[0] : nullptr;
		Tensor<Tag>* b = (params.size() >= 2) ? params[1] : nullptr;

		std::size_t w_size = w ? w->size() : 0;
		std::size_t b_size = b ? b->size() : 0;

		out.write(reinterpret_cast<const char *>(&w_size), sizeof(w_size));
		out.write(reinterpret_cast<const char *>(&b_size), sizeof(b_size));

		auto write_tensor = [&](Tensor<Tag>* t, size_t size) {
			if(size > 0 && t) {
				if constexpr(std::is_same_v<Tag, Device::CPU>) {
					out.write(reinterpret_cast<const char *>(t->data()), sizeof(float) * size);
				} else {
					Tensor<Device::CPU> temp(1, 1, 1, size);
					Tensor<Device::CPU> cpu_copy = *t;
					out.write(reinterpret_cast<const char *>(cpu_copy.data()), sizeof(float) * size);
				}
			}
		};

		write_tensor(w, w_size);
		write_tensor(b, b_size);
	}

	LOG("Model saved to", file_path);
}

template <typename Tag>
void Autoencoder<Tag>::load(const std::string &file_path) {
	std::ifstream in(file_path, std::ios::binary);
	if(!in) throw std::runtime_error("Failed to open file: " + file_path);

	std::uint32_t magic = 0, count = 0;
	in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
	in.read(reinterpret_cast<char *>(&count), sizeof(count));

	if(magic != 0x41455632) throw std::runtime_error("Model version mismatch or corrupted file");
	if(count != layers.size()) throw std::runtime_error("Layer count mismatch");

	for(auto &layer : layers) {
		std::size_t w_size_file = 0, b_size_file = 0;
		in.read(reinterpret_cast<char *>(&w_size_file), sizeof(w_size_file));
		in.read(reinterpret_cast<char *>(&b_size_file), sizeof(b_size_file));

		auto params = layer->parameters();
		Tensor<Tag>* w = (params.size() >= 1) ? params[0] : nullptr;
		Tensor<Tag>* b = (params.size() >= 2) ? params[1] : nullptr;
		
		std::size_t w_real = w ? w->size() : 0;
		std::size_t b_real = b ? b->size() : 0;

		if(w_size_file != w_real || b_size_file != b_real) {
			throw std::runtime_error("Model parameter size mismatch");
		}

		auto read_tensor = [&](Tensor<Tag>* t, size_t size) {
			if(size > 0 && t) {
				std::vector<float> buf(size);
				in.read(reinterpret_cast<char *>(buf.data()), sizeof(float) * size);
				
				if constexpr(std::is_same_v<Tag, Device::CPU>)
					std::copy(buf.begin(), buf.end(), t->data());
				else {
					Tensor<Device::CPU> temp = *t; 
					std::copy(buf.begin(), buf.end(), temp.data());
					*t = temp;
				}
			}
		};

		read_tensor(w, w_size_file);
		read_tensor(b, b_size_file);
	}
	
	LOG("Model loaded from", file_path);
}