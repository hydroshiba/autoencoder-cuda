#include "tensor.hpp"
#include "kernel.cuh"
#include "tensor.hpp"
#include "kernel.cuh"
#include <cuda_runtime.h>

namespace Autoencoder
{
    class Base;
    class CPU;
    class GPU;
}

class Layer
{
protected:
    Tensor weights, biases;
    Tensor grad_weights, grad_biases, cached_input;

    void clear_gradients();

    friend class Autoencoder::Base;
    friend class Autoencoder::CPU;
    friend class Autoencoder::GPU;

public:
    virtual Tensor forward_cpu(const Tensor &input) = 0;
    virtual Tensor forward_gpu(const Tensor &input, cudaStream_t stream = 0) = 0;

    virtual Tensor backward_cpu(const Tensor &grad_output) = 0;
    virtual Tensor backward_gpu(const Tensor &grad_output, cudaStream_t stream = 0) = 0;

    void update(float learning_rate, cudaStream_t stream = 0);
    void to_gpu();
};

class Conv2D : public Layer
{
private:
    int in_channels, out_channels;
    int kernel_size, stride, padding;

public:
    Conv2D(const int &in_channels, const int &out_channels, const int &kernel_size, const int &stride = 1, const int &padding = 0);

    Tensor forward_cpu(const Tensor &input) override;
    Tensor forward_gpu(const Tensor &input, cudaStream_t stream = 0) override;

    Tensor backward_cpu(const Tensor &grad_output) override;
    Tensor backward_gpu(const Tensor &grad_output, cudaStream_t stream = 0) override;
};

class ReLU : public Layer
{
public:
    Tensor forward_cpu(const Tensor &input) override;
    Tensor forward_gpu(const Tensor &input, cudaStream_t stream = 0) override;

    Tensor backward_cpu(const Tensor &grad_output) override;
    Tensor backward_gpu(const Tensor &grad_output, cudaStream_t stream = 0) override;
};

class MaxPool2D : public Layer
{
private:
    int pool_size;
    int stride;
    std::vector<int> max_indices; // CPU copy (for backward compatibility)
    int *d_max_indices = nullptr; // GPU-resident indices (zero-copy)
    size_t d_indices_size = 0;

public:
    MaxPool2D(int pool_size = 2, int stride = 2);
    ~MaxPool2D();

    Tensor forward_cpu(const Tensor &input) override;
    Tensor forward_gpu(const Tensor &input, cudaStream_t stream = 0) override;

    Tensor backward_cpu(const Tensor &grad_output) override;
    Tensor backward_gpu(const Tensor &grad_output, cudaStream_t stream = 0) override;
};

class UpSample2D : public Layer
{
private:
    int scale_factor;

public:
    UpSample2D(int scale = 2);

    Tensor forward_cpu(const Tensor &input) override;
    Tensor backward_cpu(const Tensor &grad_output) override;

    Tensor forward_gpu(const Tensor &input, cudaStream_t stream = 0) override;
    Tensor backward_gpu(const Tensor &grad_output, cudaStream_t stream = 0) override;
};