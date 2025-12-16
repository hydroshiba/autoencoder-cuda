#include "tensor.hpp"
#include <kernel.cuh>
#include <algorithm>
#include <cuda_runtime.h>

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this == &other)
        return *this;

    // Free current device buffer if any
    if (device_data)
    {
        CHECK(cudaFree(device_data));
        device_data = nullptr;
    }

    host_data = other.host_data;
    N = other.N;
    C = other.C;
    H = other.H;
    W = other.W;
    on_gpu = other.on_gpu;

    if (other.on_gpu)
    {
        size_t bytes = host_data.size() * sizeof(float);
        if (bytes > 0)
        {
            if (other.device_data)
            {
                CHECK(cudaMalloc(&device_data, bytes));
                CHECK(cudaMemcpy(device_data, other.device_data, bytes, cudaMemcpyDeviceToDevice));
            }
            else
            {
                // Fallback to host copy if source GPU buffer is missing
                CHECK(cudaMalloc(&device_data, bytes));
                CHECK(cudaMemcpy(device_data, host_data.data(), bytes, cudaMemcpyHostToDevice));
            }
        }
    }
    else
    {
        device_data = nullptr;
    }

    return *this;
}

void Tensor::clear_tensor()
{
    if (on_gpu && device_data)
    {
        cudaMemset(device_data, 0, host_data.size() * sizeof(float));
    }
    else
    {
        std::fill(host_data.begin(), host_data.end(), 0.0f);
    }
}

Tensor::~Tensor()
{
    if (device_data)
    {
        CHECK(cudaFree(device_data));
        device_data = nullptr;
    }
}

Tensor &Tensor::to_gpu()
{
    if (on_gpu && device_data != nullptr)
        return *this;

    on_gpu = true;

    if(host_data.size() == 0)
    {
        device_data = nullptr;
        return *this;
    }

    if (device_data)
    {
        CHECK(cudaFree(device_data));
    }

    CHECK(cudaMalloc(&device_data, host_data.size() * sizeof(float)));
    CHECK(cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    return *this;
}

Tensor &Tensor::to_cpu()
{
    if (!on_gpu)
        return *this;

    on_gpu = false;

    CHECK(cudaMemcpy(host_data.data(), device_data, host_data.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(device_data));

    device_data = nullptr;

    return *this;
}
