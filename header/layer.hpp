#include "tensor.hpp"

class Layer
{
protected:
    Tensor weights, biases;

private:
    Tensor grad_weights, grad_biases, cached_input;

public:
    virtual Tensor forward_cpu(const Tensor &input) = 0;
    virtual Tensor forward_gpu(const Tensor &input) = 0;

    virtual Tensor backward_cpu(const Tensor &grad_output) = 0;
    virtual Tensor backward_gpu(const Tensor &grad_output) = 0;

    void update(float learning_rate);
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
    Tensor forward_gpu(const Tensor &input) override;
};

class ReLU : public Layer
{
public:
    Tensor forward_cpu(const Tensor &input) override;
    Tensor forward_gpu(const Tensor &input) override;
};

class MaxPool2D : public Layer
{
private:
    int pool_size;
    int stride;
    std::vector<int> max_indices;

public:
    MaxPool2D(int pool_size = 2, int stride = 2);

    Tensor forward_cpu(const Tensor &input) override;
    Tensor backward_cpu(const Tensor &grad_output) override;

    Tensor forward_gpu(const Tensor &input) override;
    Tensor backward_gpu(const Tensor &grad_output) override;
};

class UpSample2D : public Layer
{
private:
    int scale_factor;

public:
    UpSample2D(int scale = 2);

    Tensor forward_cpu(const Tensor &input) override;
    Tensor backward_cpu(const Tensor &grad_output) override;

    Tensor forward_gpu(const Tensor &input) override;
    Tensor backward_gpu(const Tensor &grad_output) override;
};