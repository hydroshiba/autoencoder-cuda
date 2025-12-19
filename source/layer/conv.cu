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

// 2D Convolution kernel: processes output elements in parallel.
// Each thread computes one output element.
__global__ void conv2d_forward_kernel(
    const float *input,   // shape: (batch, in_channels, input_h, input_w)
    const float *weights, // shape: (out_channels, in_channels, kernel_size, kernel_size)
    const float *biases,  // shape: (out_channels,)
    float *output,        // shape: (batch, out_channels, output_h, output_w)
    int batch, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_size, int stride, int padding,
    int output_h, int output_w)
{
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * out_channels * output_h * output_w;

    if (output_idx >= total_output)
        return;

    // Decompose linear index into (n, oc, oh, ow)
    int n = output_idx / (out_channels * output_h * output_w);
    int remainder = output_idx % (out_channels * output_h * output_w);
    int oc = remainder / (output_h * output_w);
    remainder = remainder % (output_h * output_w);
    int oh = remainder / output_w;
    int ow = remainder % output_w;

    float sum = biases[oc];

    // Convolution: iterate over all input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic)
    {
        for (int kh = 0; kh < kernel_size; ++kh)
        {
            for (int kw = 0; kw < kernel_size; ++kw)
            {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                // Skip padding regions
                if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w)
                    continue;

                // Compute flat indices in the linearized arrays
                int input_idx = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
                int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    output[output_idx] = sum;
}

// Backward kernels for Conv2D
__global__ void conv2d_bias_grad_kernel(const float *grad_out, float *grad_b, int N, int OC, int H_out, int W_out)
{
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC)
        return;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        for (int oh = 0; oh < H_out; ++oh)
            for (int ow = 0; ow < W_out; ++ow)
                sum += grad_out[((n * OC + oc) * H_out + oh) * W_out + ow];
    grad_b[oc] = sum;
}

__global__ void conv2d_weight_grad_kernel(const float *input, const float *grad_out, float *grad_w,
                                          int N, int IC, int OC, int H_in, int W_in,
                                          int K, int stride, int pad, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OC * IC * K * K;
    if (idx >= total)
        return;
    int oc = idx / (IC * K * K);
    int rem = idx % (IC * K * K);
    int ic = rem / (K * K);
    rem = rem % (K * K);
    int kh = rem / K;
    int kw = rem % K;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        for (int oh = 0; oh < H_out; ++oh)
        {
            for (int ow = 0; ow < W_out; ++ow)
            {
                int ih = oh * stride + kh - pad;
                int iw = ow * stride + kw - pad;
                if (ih < 0 || ih >= H_in || iw < 0 || iw >= W_in)
                    continue;
                int in_idx = ((n * IC + ic) * H_in + ih) * W_in + iw;
                int go_idx = ((n * OC + oc) * H_out + oh) * W_out + ow;
                sum += input[in_idx] * grad_out[go_idx];
            }
        }
    }
    grad_w[idx] = sum;
}

__global__ void conv2d_input_grad_kernel(const float *grad_out, const float *weights, float *grad_in,
                                         int N, int IC, int OC, int H_in, int W_in,
                                         int K, int stride, int pad, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * IC * H_in * W_in;
    if (idx >= total)
        return;
    int n = idx / (IC * H_in * W_in);
    int rem = idx % (IC * H_in * W_in);
    int ic = rem / (H_in * W_in);
    rem = rem % (H_in * W_in);
    int ih = rem / W_in;
    int iw = rem % W_in;

    float sum = 0.0f;
    for (int oc = 0; oc < OC; ++oc)
    {
        for (int kh = 0; kh < K; ++kh)
        {
            for (int kw = 0; kw < K; ++kw)
            {
                int oh_unstrided = ih + pad - kh;
                int ow_unstrided = iw + pad - kw;
                if (oh_unstrided < 0 || ow_unstrided < 0)
                    continue;
                if (oh_unstrided % stride != 0 || ow_unstrided % stride != 0)
                    continue;
                int oh = oh_unstrided / stride;
                int ow = ow_unstrided / stride;
                if (oh < 0 || oh >= H_out || ow < 0 || ow >= W_out)
                    continue;

                int go_idx = ((n * OC + oc) * H_out + oh) * W_out + ow;
                int w_idx = ((oc * IC + ic) * K + kh) * K + kw;
                sum += grad_out[go_idx] * weights[w_idx];
            }
        }
    }
    grad_in[idx] = sum;
}

Tensor Conv2D::forward_gpu(const Tensor &input, cudaStream_t stream)
{
    if (input.channels() != in_channels)
    {
        throw std::invalid_argument("Conv2D forward GPU: input channel mismatch");
    }

    cached_input = input;

    const int batch_size = input.batch();
    const int input_height = input.height();
    const int input_width = input.width();

    if (input_height <= 0 || input_width <= 0)
    {
        throw std::invalid_argument("Conv2D forward GPU: invalid spatial dimensions");
    }

    const int numerator_h = input_height + 2 * padding - kernel_size;
    const int numerator_w = input_width + 2 * padding - kernel_size;

    if (numerator_h < 0 || numerator_w < 0)
    {
        throw std::invalid_argument("Conv2D forward GPU: configuration yields negative output dimensions");
    }

    if (numerator_h % stride != 0 || numerator_w % stride != 0)
    {
        throw std::invalid_argument("Conv2D forward GPU: stride does not divide padded dimensions");
    }

    const int output_height = numerator_h / stride + 1;
    const int output_width = numerator_w / stride + 1;
    const size_t output_size = batch_size * out_channels * output_height * output_width;

    // Ensure input is on GPU
    Tensor input_gpu = input;
    if (!input_gpu.is_gpu())
    {
        input_gpu.to_gpu();
    }

    // Ensure weights and biases are on GPU
    if (!weights.is_gpu())
    {
        weights.to_gpu();
    }
    if (!biases.is_gpu())
    {
        biases.to_gpu();
    }

    // Create output on GPU
    Tensor output(batch_size, out_channels, output_height, output_width, true);
    output.to_gpu();

    // Configure kernel launch
    const int threads_per_block = 256;
    const int blocks = (output_size + threads_per_block - 1) / threads_per_block;

    // Launch kernel using device pointers directly with stream
    conv2d_forward_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input_gpu.data(), weights.data(), biases.data(), output.data(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding,
        output_height, output_width);

    CUDA_CHECK(cudaGetLastError());

    return output;
}

Tensor Conv2D::backward_gpu(const Tensor &grad_output, cudaStream_t stream)
{
    const int N = cached_input.batch();
    const int C_in = in_channels;
    const int H_in = cached_input.height();
    const int W_in = cached_input.width();
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    const size_t in_size = static_cast<size_t>(N) * C_in * H_in * W_in;
    const size_t out_size = static_cast<size_t>(N) * out_channels * H_out * W_out;
    const size_t w_size = static_cast<size_t>(out_channels) * C_in * kernel_size * kernel_size;
    const size_t b_size = static_cast<size_t>(out_channels);

    // Ensure cached_input and grad_output are on GPU
    Tensor cached_input_gpu = cached_input;
    if (!cached_input_gpu.is_gpu())
    {
        cached_input_gpu.to_gpu();
    }

    Tensor grad_output_gpu = grad_output;
    if (!grad_output_gpu.is_gpu())
    {
        grad_output_gpu.to_gpu();
    }

    // Ensure weights on GPU
    if (!weights.is_gpu())
    {
        weights.to_gpu();
    }

    // Ensure grad tensors are on GPU
    if (!grad_weights.is_gpu())
    {
        grad_weights.to_gpu();
    }
    if (!grad_biases.is_gpu())
    {
        grad_biases.to_gpu();
    }

    // Create grad_input on GPU
    Tensor grad_input(N, C_in, H_in, W_in, true);
    grad_input.to_gpu();

    // Zero gradients on device
    CUDA_CHECK(cudaMemsetAsync(grad_input.data(), 0, in_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(grad_weights.data(), 0, w_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(grad_biases.data(), 0, b_size * sizeof(float), stream));

    // Launch kernels using device pointers directly with stream
    int threads = 256;
    int blocks_b = (out_channels + threads - 1) / threads;
    conv2d_bias_grad_kernel<<<blocks_b, threads, 0, stream>>>(
        grad_output_gpu.data(), grad_biases.data(),
        N, out_channels, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());

    int total_w = static_cast<int>(w_size);
    int blocks_w = (total_w + threads - 1) / threads;
    conv2d_weight_grad_kernel<<<blocks_w, threads, 0, stream>>>(
        cached_input_gpu.data(), grad_output_gpu.data(), grad_weights.data(),
        N, C_in, out_channels, H_in, W_in, kernel_size, stride, padding, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());

    int total_in = static_cast<int>(in_size);
    int blocks_in = (total_in + threads - 1) / threads;
    conv2d_input_grad_kernel<<<blocks_in, threads, 0, stream>>>(
        grad_output_gpu.data(), weights.data(), grad_input.data(),
        N, C_in, out_channels, H_in, W_in, kernel_size, stride, padding, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());

    return grad_input;
}