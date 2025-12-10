#include "autoencoder.hpp"

using namespace Autoencoder;

void CPU::update(float learning_rate) {
	for (const auto& layer : layers) {
		layer->update(learning_rate);
	}
}