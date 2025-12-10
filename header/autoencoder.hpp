#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include <vector>
#include <memory>

#include "layer.hpp"

namespace Autoencoder {

class Base {
protected:
	std::vector<std::unique_ptr<Layer>> layers;
	int encode_layer = 0;

	void add_layer(std::unique_ptr<Layer> layer);
	
	virtual Tensor forward_encode(const Tensor &input) = 0;
	virtual Tensor forward_decode(const Tensor &input) = 0;

	virtual Tensor backward_decode(const Tensor &gradient) = 0;
	virtual Tensor backward_encode(const Tensor &gradient) = 0;

public:
	Base();
	void build();

	Tensor forward(const Tensor &input);
	Tensor encode(const Tensor &input);
	void backward(const Tensor &gradient);
};

class CPU: public Base {
public:
	void update(float learning_rate);
};

class GPU: public Base {
public:
	GPU();
	GPU(const Base &base);
	void update(float learning_rate);
};

}

#endif