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

__global__ void relu_backward_kernel(const float *input, const float *grad_output, float *grad_input, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
	}
}

Tensor ReLU::forward_gpu(const Tensor &input, cudaStream_t stream)
{
	cached_input = input;

	// Ensure input is on GPU
	Tensor input_gpu = input;
	if (!input_gpu.is_gpu())
	{
		input_gpu.to_gpu();
	}

	// Create output on GPU
	Tensor output(input.batch(), input.channels(), input.height(), input.width(), true);
	output.to_gpu();

	dim3 blockSize(256);
	dim3 gridSize((output.size() + blockSize.x - 1) / blockSize.x);
	relu_forward_kernel<<<gridSize, blockSize, 0, stream>>>(input_gpu.data(), output.data(), output.size());

	CHECK(cudaGetLastError());

	return output;
}

Tensor ReLU::backward_gpu(const Tensor &grad_output, cudaStream_t stream)
{
	// Ensure grad_output is on GPU
	Tensor grad_output_gpu = grad_output;
	if (!grad_output_gpu.is_gpu())
	{
		grad_output_gpu.to_gpu();
	}

	// Ensure cached_input is on GPU
	Tensor cached_on_gpu = cached_input;
	if (!cached_on_gpu.is_gpu())
	{
		cached_on_gpu.to_gpu();
	}

	// Create grad_input on GPU
	Tensor grad_input(grad_output.batch(), grad_output.channels(), grad_output.height(), grad_output.width(), true);
	grad_input.to_gpu();

	dim3 blockSize(256);
	dim3 gridSize((grad_input.size() + blockSize.x - 1) / blockSize.x);
	relu_backward_kernel<<<gridSize, blockSize, 0, stream>>>(cached_on_gpu.data(), grad_output_gpu.data(), grad_input.data(), grad_input.size());

	CHECK(cudaGetLastError());

	return grad_input;
}
