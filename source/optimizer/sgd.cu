#include "kernel.cuh"

#include <stdexcept>
#include <iostream>

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

void sgd_update_device(float *params, const float *grads, float lr, std::size_t size, cudaStream_t stream)
{
    if (!params || !grads || size == 0)
    {
        std::cout << "[sgd_update_device] Skip: params=" << params << ", grads=" << grads << ", size=" << size << std::endl;
        return; // nothing to do or invalid pointers
    }

    const int threads = 256;
    const int blocks = static_cast<int>((size + threads - 1) / threads);

    std::cout << "[sgd_update_device] Launch kernel: size=" << size << ", blocks=" << blocks << ", threads=" << threads << std::endl;
    sgd_update_kernel<<<blocks, threads, 0, stream>>>(params, grads, lr, static_cast<int>(size));
    std::cout << "[sgd_update_device] Checking last error..." << std::endl;
    CUDA_CHECK(cudaGetLastError());
    std::cout << "[sgd_update_device] Done" << std::endl;
}
