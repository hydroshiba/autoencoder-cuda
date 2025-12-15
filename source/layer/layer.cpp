#include "layer.hpp"

void Layer::to_gpu()
{
    weights.to_gpu();
    biases.to_gpu();
    grad_weights.to_gpu();
    grad_biases.to_gpu();
    cached_input.to_gpu();
}

void Layer::update(float learning_rate)
{
    // Update weights and biases using gradients; dispatch to GPU when tensors reside on device
    const bool use_gpu = weights.is_gpu() && grad_weights.is_gpu() && biases.is_gpu() && grad_biases.is_gpu();

    if (use_gpu)
    {
        sgd_update_device(weights.data(), grad_weights.data(), learning_rate, weights.size());
        sgd_update_device(biases.data(), grad_biases.data(), learning_rate, biases.size());
    }
    else
    {
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            weights.data()[i] -= learning_rate * grad_weights.data()[i];
        }
        for (std::size_t i = 0; i < biases.size(); ++i)
        {
            biases.data()[i] -= learning_rate * grad_biases.data()[i];
        }
    }
}

void Layer::clear_gradients()
{
    grad_weights.clear_tensor();
    grad_biases.clear_tensor();
}