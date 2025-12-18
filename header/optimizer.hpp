#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"
#include "autoencoder.hpp"

namespace Optimizer {

class SGD {

public:
	SGD() = default;

	template <typename Model>
	void step(Model &model, float learning_rate) {
		auto &layers = model.get_layers();

		for(auto &layer : layers) {
			auto params = layer->parameters();
			auto grads = layer->gradients();

			if(params.empty() || params.size() != grads.size()) continue;

			for(size_t i = 0; i < params.size(); ++i) {
				if(params[i] && grads[i]) {
					*grads[i] *= learning_rate;
					*params[i] -= *grads[i];
				}
			}
		}
	}
};

}

#endif // OPTIMIZER_HPP