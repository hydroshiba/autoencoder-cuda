#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include <vector>
#include <memory>
#include <string>

#include "layer.hpp"
#include "config.hpp"

template <typename Tag>
class Autoencoder {
	static_assert(Device::IsTag_v<Tag>, "Autoencoder tag must be Device::CPU or Device::GPU");

private:
	std::vector<std::unique_ptr<Layer::Base<Tag>>> layers;
	int encode_layer = 0;

	// Internal helpers
	Tensor<Tag> forward_encode(const Tensor<Tag> &input);
	Tensor<Tag> forward_decode(const Tensor<Tag> &input);
	
	Tensor<Tag> backward_encode(const Tensor<Tag> &gradient);
	Tensor<Tag> backward_decode(const Tensor<Tag> &gradient);

public:
	Autoencoder();
	Autoencoder(const Autoencoder &other);
	Autoencoder& operator=(const Autoencoder &other);
	Autoencoder(Autoencoder&& other) = default;
	
	void build();
	void clear_gradients();

	// Forward / Backward API
	Tensor<Tag> forward(const Tensor<Tag> &input);
	Tensor<Tag> encode(const Tensor<Tag> &input);
	Tensor<Tag> decode(const Tensor<Tag> &input);
	void backward(const Tensor<Tag> &gradient);

	// Expose layers for the Optimizer
	std::vector<std::unique_ptr<Layer::Base<Tag>>>& get_layers();

	// IO
	void save(const std::string &file_path);
	void load(const std::string &file_path);
};

#include "autoencoder.tpp"

#endif // AUTOENCODER_HPP