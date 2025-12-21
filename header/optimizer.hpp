#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"

namespace Optimizer {

class SGD {

public:
	SGD() = default;

	template <typename Model>
	void step(Model &model, float learning_rate) {
		auto &layers = model.get_layers();
		const float clip_limit = 1.0f;

		for(auto &layer : layers) {
			auto params = layer->parameters();
			auto grads = layer->gradients();
			if(params.empty() || params.size() != grads.size()) continue;

			for(size_t i = 0; i < params.size(); ++i) if(params[i] && grads[i]) {
				grads[i]->transform([clip_limit] __host__ __device__ (float g) {
					if(g > clip_limit) return clip_limit;
					if(g < -clip_limit) return -clip_limit;
					return g;
				});
				*params[i] -= *grads[i] * learning_rate;
			}
		}
	}
};

}

#endif // OPTIMIZER_HPP