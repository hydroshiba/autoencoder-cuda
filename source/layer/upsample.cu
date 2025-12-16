#include "layer.hpp"
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

// Nearest-neighbor upsample forward kernel
__global__ void upsample_forward_kernel(const float *input, float *output,
                                        int N, int C, int H, int W, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outH = H * scale;
    int outW = W * scale;
    int total = N * C * outH * outW;
    if (idx >= total)
        return;

    int n = idx / (C * outH * outW);
    int rem = idx % (C * outH * outW);
    int c = rem / (outH * outW);
    rem = rem % (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;

    int ih = oh / scale;
    int iw = ow / scale;

    int in_idx = ((n * C + c) * H + ih) * W + iw;
    output[idx] = input[in_idx];
}

// Backward kernel: accumulate gradients from expanded output back to input
__global__ void upsample_backward_kernel(const float *grad_out, float *grad_in,
                                         int N, int C, int H, int W, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total)
        return;

    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int ih = rem / W;
    int iw = rem % W;

    int outH = H * scale;
    int outW = W * scale;

    float sum = 0.0f;
    int oh_start = ih * scale;
    int ow_start = iw * scale;
    for (int dh = 0; dh < scale; ++dh)
    {
        for (int dw = 0; dw < scale; ++dw)
        {
            int oh = oh_start + dh;
            int ow = ow_start + dw;
            int out_idx = ((n * C + c) * outH + oh) * outW + ow;
            sum += grad_out[out_idx];
        }
    }
    grad_in[idx] = sum;
}

Tensor UpSample2D::forward_gpu(const Tensor &input)
{
    cached_input = input;
    const int N = input.batch();
    const int C = input.channels();
    const int H = input.height();
    const int W = input.width();
    if (H <= 0 || W <= 0 || C <= 0 || N <= 0)
        throw std::invalid_argument("UpSample2D forward GPU: invalid input dims");

    const int outH = H * scale_factor;
    const int outW = W * scale_factor;
    const size_t out_size = static_cast<size_t>(N) * C * outH * outW;

    // Ensure input is on GPU
    Tensor input_gpu = input;
    if (!input_gpu.is_gpu())
    {
        input_gpu.to_gpu();
    }

    // Create output on GPU
    Tensor output(N, C, outH, outW, true);
    output.to_gpu();

    int threads = 256;
    int blocks = static_cast<int>((out_size + threads - 1) / threads);
    upsample_forward_kernel<<<blocks, threads>>>(
        input_gpu.data(), output.data(), N, C, H, W, scale_factor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

Tensor UpSample2D::backward_gpu(const Tensor &grad_output)
{
    const int N = cached_input.batch();
    const int C = cached_input.channels();
    const int H = cached_input.height();
    const int W = cached_input.width();
    const int outH = H * scale_factor;
    const int outW = W * scale_factor;

    const size_t in_size = static_cast<size_t>(N) * C * H * W;

    // Ensure grad_output is on GPU
    Tensor grad_output_gpu = grad_output;
    if (!grad_output_gpu.is_gpu())
    {
        grad_output_gpu.to_gpu();
    }

    // Create grad_input on GPU
    Tensor grad_input(N, C, H, W, true);
    grad_input.to_gpu();
    CUDA_CHECK(cudaMemset(grad_input.data(), 0, in_size * sizeof(float)));

    int threads = 256;
    int blocks = static_cast<int>((in_size + threads - 1) / threads);
    upsample_backward_kernel<<<blocks, threads>>>(
        grad_output_gpu.data(), grad_input.data(), N, C, H, W, scale_factor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}