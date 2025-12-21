#include "layer.hpp"
#include "utils/error.cuh"
#include "utils/kernel.cuh"

// Max Pooling 2D GPU Kernels

__global__ void maxpool_forward_kernel(
	const float *input, float *output, float *mask,
	int N, int C, int H, int W, 
	int pool, int out_h, int out_w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * C * out_h * out_w;
	if(idx >= total) return;

	int n = idx / (C * out_h * out_w);
	int rem = idx % (C * out_h * out_w);
	int c = rem / (out_h * out_w);
	rem = rem % (out_h * out_w);
	int oh = rem / out_w;
	int ow = rem % out_w;

	float max_val = -1e9f;
	int max_local = 0;

	for(int kh = 0; kh < pool; ++kh) {
		for(int kw = 0; kw < pool; ++kw) {
			int ih = oh * pool + kh;
			int iw = ow * pool + kw;
			
			int flat = ((n * C + c) * H + ih) * W + iw;
			float v = input[flat];
			
			if(v > max_val) {
				max_val = v;
				// Store relative local index (e.g. 0, 1, 2, 3)
				max_local = kh * pool + kw;
			}
		}
	}

	output[idx] = max_val;
	// Store local index as float (Safe for small integers)
	mask[idx] = static_cast<float>(max_local);
}

__global__ void maxpool_backward_kernel(
	const float *grad_output, float *grad_input, const float *mask, 
	int total, int C, int H, int W, 
	int pool, int out_h, int out_w) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= total) return;
	
	// Reconstruct coordinates
	int n = idx / (C * out_h * out_w);
	int rem = idx % (C * out_h * out_w);
	int c = rem / (out_h * out_w);
	rem = rem % (out_h * out_w);
	int oh = rem / out_w;
	int ow = rem % out_w;

	// Retrieve local offset and calculate input coordinates
	int local_idx = static_cast<int>(mask[idx]);
	int kh = local_idx / pool;
	int kw = local_idx % pool;

	int ih = oh * pool + kh;
	int iw = ow * pool + kw;

	int in_idx = ((n * C + c) * H + ih) * W + iw;
	
	// Atomic add is safer if there's any overlap, though standard maxpool usually implies 1:1 mapping here
	atomicAdd(&grad_input[in_idx], grad_output[idx]);
}

// Max Pooling 2D GPU specialization implementations

namespace Layer {

template <>
Tensor<Device::GPU> MaxPool2D<Device::GPU>::forward(const Tensor<Device::GPU> &input) {
	this->cached_input = input;

	int N = input.batches();
	int C = input.channels();
	int H = input.height();
	int W = input.width();

	int out_h = H / pool_size;
	int out_w = W / pool_size;

	Tensor<Device::GPU> output(N, C, out_h, out_w);
	this->mask = Tensor<Device::GPU>(N, C, out_h, out_w);

	size_t out_size = output.size();
	int threads = Config::MaxPool2D::block_width * Config::MaxPool2D::block_height;
	int blocks = (out_size + threads - 1) / threads;

	maxpool_forward_kernel<<<blocks, threads>>>(
		input.data(), output.data(), mask.data(),
		N, C, H, W, pool_size, out_h, out_w
	);
	
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	this->cached_output = output;
	return output;
}

template <>
Tensor<Device::GPU> MaxPool2D<Device::GPU>::backward(const Tensor<Device::GPU> &grad_output) {
	int N = cached_input.batches();
	int C = cached_input.channels();
	int H = cached_input.height();
	int W = cached_input.width();

	int out_h = H / pool_size;
	int out_w = W / pool_size;

	Tensor<Device::GPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);

	Tensor<Device::GPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);
	
	size_t total = derivatives.size();
	int threads_mul = Config::Tensor::block_width * Config::Tensor::block_height;
	int blocks_mul = (total + threads_mul - 1) / threads_mul;
	Kernel::vector_multiply<<<blocks_mul, threads_mul>>>(derivatives.data(), grad_output.data(), total);
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	size_t out_size = grad_output.size();
	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	int blocks = (out_size + threads - 1) / threads;

	maxpool_backward_kernel<<<blocks, threads>>>(
		derivatives.data(), grad_input.data(), mask.data(), 
		static_cast<int>(out_size), C, H, W, 
		pool_size, out_h, out_w
	);

	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	return grad_input;
}

}