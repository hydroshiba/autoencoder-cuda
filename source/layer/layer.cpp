#include "layer.hpp"

void Layer::to_gpu() {
    weights.to_gpu();
    biases.to_gpu();
    grad_weights.to_gpu();
    grad_biases.to_gpu();
    cached_input.to_gpu();
}

void Layer::update(float learning_rate) {
    // Update weights and biases using gradients
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights.host_data[i] -= learning_rate * grad_weights.host_data[i];
    }
    for (std::size_t i = 0; i < biases.size(); ++i) {
        biases.host_data[i] -= learning_rate * grad_biases.host_data[i];
    }
}