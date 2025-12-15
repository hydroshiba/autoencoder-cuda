#include "kernel.cuh"

#include <stdexcept>

#define CUDA_CHECK(err)                                                                       \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err_ = (err);                                                             \
        if (err_ != cudaSuccess)                                                              \
        {                                                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_)); \
        }                                                                                     \
    } while (0)

__global__ void sgd_update_kernel(float *params, const float *grads, float lr, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        params[i] -= lr * grads[i];
    }
}

void sgd_update_device(float *params, const float *grads, float lr, std::size_t size)
{
    if (!params || !grads || size == 0)
    {
        return; // nothing to do or invalid pointers
    }

    const int threads = 256;
    const int blocks = static_cast<int>((size + threads - 1) / threads);

    sgd_update_kernel<<<blocks, threads>>>(params, grads, lr, static_cast<int>(size));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
