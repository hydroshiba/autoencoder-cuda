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

__global__ void maxpool_forward_kernel(const float *input, float *output, int *max_indices,
									   int N, int C, int H, int W, int pool, int out_h, int out_w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * C * out_h * out_w;
	if (idx >= total)
		return;

	int n = idx / (C * out_h * out_w);
	int rem = idx % (C * out_h * out_w);
	int c = rem / (out_h * out_w);
	rem = rem % (out_h * out_w);
	int oh = rem / out_w;
	int ow = rem % out_w;

	float max_val = -1e9f;
	int max_flat = -1;

	for (int kh = 0; kh < pool; ++kh)
	{
		for (int kw = 0; kw < pool; ++kw)
		{
			int ih = oh * pool + kh;
			int iw = ow * pool + kw;
			int flat = ((n * C + c) * H + ih) * W + iw;
			float v = input[flat];
			if (v > max_val)
			{
				max_val = v;
				max_flat = flat;
			}
		}
	}

	output[idx] = max_val;
	max_indices[idx] = max_flat;
}

__global__ void maxpool_backward_kernel(const float *grad_output, float *grad_input, const int *max_indices, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total)
		return;
	int in_idx = max_indices[idx];
	atomicAdd(&grad_input[in_idx], grad_output[idx]);
}

Tensor MaxPool2D::forward_gpu(const Tensor &input)
{
	cached_input = input;

	const int N = input.batch();
	const int C = input.channels();
	const int H = input.height();
	const int W = input.width();
	const int out_h = H / pool_size;
	const int out_w = W / pool_size;

	const size_t out_size = static_cast<size_t>(N) * C * out_h * out_w;

	// Ensure input is on GPU
	Tensor input_gpu = input;
	if (!input_gpu.is_gpu())
	{
		input_gpu.to_gpu();
	}

	// Create output on GPU
	Tensor output(N, C, out_h, out_w, true);
	output.to_gpu();

	// Allocate device memory for max indices
	max_indices.resize(out_size);
	int *d_indices = nullptr;
	CUDA_CHECK(cudaMalloc(&d_indices, out_size * sizeof(int)));

	const int threads = 256;
	const int blocks = static_cast<int>((out_size + threads - 1) / threads);
	maxpool_forward_kernel<<<blocks, threads>>>(
		input_gpu.data(), output.data(), d_indices,
		N, C, H, W, pool_size, out_h, out_w);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Copy max_indices back to host (needed for backward pass)
	CUDA_CHECK(cudaMemcpy(max_indices.data(), d_indices, out_size * sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(d_indices);

	return output;
}

Tensor MaxPool2D::backward_gpu(const Tensor &grad_output)
{
	const int N = cached_input.batch();
	const int C = cached_input.channels();
	const int H = cached_input.height();
	const int W = cached_input.width();
	const int out_h = H / pool_size;
	const int out_w = W / pool_size;

	const size_t in_size = static_cast<size_t>(N) * C * H * W;
	const size_t out_size = static_cast<size_t>(N) * C * out_h * out_w;

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

	// Upload max_indices to device
	int *d_indices = nullptr;
	CUDA_CHECK(cudaMalloc(&d_indices, out_size * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_indices, max_indices.data(), out_size * sizeof(int), cudaMemcpyHostToDevice));

	const int threads = 256;
	const int blocks = static_cast<int>((out_size + threads - 1) / threads);
	maxpool_backward_kernel<<<blocks, threads>>>(
		grad_output_gpu.data(), grad_input.data(), d_indices, static_cast<int>(out_size));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaFree(d_indices);

	return grad_input;
}