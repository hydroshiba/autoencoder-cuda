#include "autoencoder.hpp"

using namespace Autoencoder;

GPU::GPU() : Base() {
	for(auto& layer: layers) {
		layer = layer->to_gpu();
	}
}

GPU::GPU(const Base &base) : Base(base) {
	for(int i = 0; i < layers.size(); i++) {
		layers[i].get() = base.layers[i].get();
		layers[i]->to_gpu();
	}
}