#include "layer.hpp"

#include <stdexcept>

#define CHECK(err)                                                                            \
	do                                                                                        \
	{                                                                                         \
		cudaError_t err_ = (err);                                                             \
		if (err_ != cudaSuccess)                                                              \
		{                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_)); \
		}                                                                                     \
	} while (0)

__global__ void relu_forward_kernel(const float *input, float *output, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		output[i] = input[i] > 0.f ? input[i] : 0.f;
	}
}

// ReLU backward kernel: pass gradients only where input > 0
__global__ void relu_backward_kernel(const float *input, float *grad, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		if (input[i] <= 0.0f)
		{
			grad[i] = 0.0f;
		}
	}
}

Tensor ReLU::forward_gpu(const Tensor &input)
{
	cached_input = input;
	Tensor output(input);
	output = output.to_gpu();

	dim3 blockSize(256);
	dim3 gridSize((output.size() + blockSize.x - 1) / blockSize.x);
	relu_forward_kernel<<<gridSize, blockSize>>>(output.data(), output.data(), output.size());

	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());

	return output;
}

Tensor ReLU::backward_gpu(const Tensor &grad_output)
{
	Tensor grad_input(grad_output);
	grad_input = grad_input.to_gpu();

	// Ensure cached_input is on device for kernel access
	Tensor cached_on_gpu = cached_input;
	if (!cached_on_gpu.is_gpu())
	{
		cached_on_gpu.to_gpu();
	}

	dim3 blockSize(256);
	dim3 gridSize((grad_input.size() + blockSize.x - 1) / blockSize.x);
	relu_backward_kernel<<<gridSize, blockSize>>>(cached_on_gpu.data(), grad_input.data(), grad_input.size());

	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());

	return grad_input;
}
