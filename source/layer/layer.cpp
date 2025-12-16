#include "layer.hpp"
#include <iostream>

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
    // Skip update if no parameters to update
    if (weights.size() == 0 && biases.size() == 0)
    {
        return;
    }

    // Update weights and biases using gradients; dispatch to GPU when tensors reside on device
    const bool use_gpu = weights.is_gpu() && grad_weights.is_gpu() && biases.is_gpu() && grad_biases.is_gpu();

    if (use_gpu)
    {
        std::cout << "[Layer::update] GPU mode, weights.size=" << weights.size() << ", biases.size=" << biases.size() << std::endl;
        if (weights.size() > 0)
        {
            std::cout << "[Layer::update] Updating weights..." << std::endl;
            sgd_update_device(weights.data(), grad_weights.data(), learning_rate, weights.size());
            std::cout << "[Layer::update] Weights updated" << std::endl;
        }
        if (biases.size() > 0)
        {
            std::cout << "[Layer::update] Updating biases..." << std::endl;
            sgd_update_device(biases.data(), grad_biases.data(), learning_rate, biases.size());
            std::cout << "[Layer::update] Biases updated" << std::endl;
        }
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