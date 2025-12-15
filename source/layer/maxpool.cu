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

	Tensor output(N, C, out_h, out_w);
	max_indices.resize(N * C * out_h * out_w);

	const size_t in_size = static_cast<size_t>(N) * C * H * W;
	const size_t out_size = static_cast<size_t>(N) * C * out_h * out_w;

	float *d_input = nullptr, *d_output = nullptr;
	int *d_indices = nullptr;

	CUDA_CHECK(cudaMalloc(&d_input, in_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_output, out_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_indices, out_size * sizeof(int)));

	try
	{
		CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));

		const int threads = 256;
		const int blocks = static_cast<int>((out_size + threads - 1) / threads);
		maxpool_forward_kernel<<<blocks, threads>>>(d_input, d_output, d_indices, N, C, H, W, pool_size, out_h, out_w);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(max_indices.data(), d_indices, out_size * sizeof(int), cudaMemcpyDeviceToHost));

		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_indices);
	}
	catch (...)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_indices);
		throw;
	}

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

	Tensor grad_input(N, C, H, W);
	std::fill(grad_input.data(), grad_input.data() + grad_input.size(), 0.0f);

	const size_t in_size = static_cast<size_t>(N) * C * H * W;
	const size_t out_size = static_cast<size_t>(N) * C * out_h * out_w;

	float *d_grad_out = nullptr, *d_grad_in = nullptr;
	int *d_indices = nullptr;

	CUDA_CHECK(cudaMalloc(&d_grad_out, out_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_grad_in, in_size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_indices, out_size * sizeof(int)));

	try
	{
		CUDA_CHECK(cudaMemcpy(d_grad_out, grad_output.data(), out_size * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_grad_in, grad_input.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_indices, max_indices.data(), out_size * sizeof(int), cudaMemcpyHostToDevice));

		const int threads = 256;
		const int blocks = static_cast<int>((out_size + threads - 1) / threads);
		maxpool_backward_kernel<<<blocks, threads>>>(d_grad_out, d_grad_in, d_indices, static_cast<int>(out_size));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(grad_input.data(), d_grad_in, in_size * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(d_grad_out);
		cudaFree(d_grad_in);
		cudaFree(d_indices);
	}
	catch (...)
	{
		cudaFree(d_grad_out);
		cudaFree(d_grad_in);
		cudaFree(d_indices);
		throw;
	}

	return grad_input;
}