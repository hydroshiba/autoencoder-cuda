#include "layer.hpp"

Tensor UpSample2D::forward_gpu(const Tensor &input)
{
    // GPU version: currently fallback to CPU
    return forward_cpu(input);
}

Tensor UpSample2D::backward_gpu(const Tensor &grad_output)
{
    // GPU version: currently fallback to CPU
    return backward_cpu(grad_output);
}