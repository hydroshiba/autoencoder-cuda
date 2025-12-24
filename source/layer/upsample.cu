#include "layer.hpp"
#include "utils/error.cuh"
#include "utils/kernel.cuh"

// Upsample 2D GPU Kernels

__global__ void upsample_forward_kernel(
	const float *input, float *output,
	int N, int C, int H, int W, int scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int outH = H * scale;
	int outW = W * scale;
	int total = N * C * outH * outW;
	
	if(idx >= total) return;

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

__global__ void upsample_backward_kernel(
	const float *grad_out, float *grad_in,
	int N, int C, int H, int W, int scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * C * H * W;
	
	if(idx >= total) return;

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
	
	for(int dh = 0; dh < scale; ++dh) {
		for(int dw = 0; dw < scale; ++dw) {
			int oh = oh_start + dh;
			int ow = ow_start + dw;
			int out_idx = ((n * C + c) * outH + oh) * outW + ow;
			sum += grad_out[out_idx];
		}
	}

	grad_in[idx] = sum;
}

// Upsample 2D GPU specialization implementations

namespace Layer {

template <>
Tensor<Device::GPU> UpSample2D<Device::GPU>::forward(const Tensor<Device::GPU> &input) {
	this->cached_input = input;

	int N = input.batches();
	int C = input.channels();
	int H = input.height();
	int W = input.width();

	int out_h = H * scale;
	int out_w = W * scale;
	size_t out_size = static_cast<size_t>(N) * C * out_h * out_w;

	Tensor<Device::GPU> output(N, C, out_h, out_w);

	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	int blocks = (out_size + threads - 1) / threads;

	upsample_forward_kernel<<<blocks, threads>>>(
		input.data(), output.data(), N, C, H, W, scale
	);

	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	this->cached_output = output;
	return output;
}

template <>
Tensor<Device::GPU> UpSample2D<Device::GPU>::backward(const Tensor<Device::GPU> &grad_output) {
	int N = cached_input.batches();
	int C = cached_input.channels();
	int H = cached_input.height();
	int W = cached_input.width();

	size_t in_size = static_cast<size_t>(N) * C * H * W;

	Tensor<Device::GPU> grad_input(N, C, H, W);

	Tensor<Device::GPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);
	
	size_t total = derivatives.size();
	int threads_mul = Config::Tensor::block_width * Config::Tensor::block_height;
	int blocks_mul = (total + threads_mul - 1) / threads_mul;
	Kernel::vector_multiply<<<blocks_mul, threads_mul>>>(derivatives.data(), grad_output.data(), total);
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	int blocks = (in_size + threads - 1) / threads;

	upsample_backward_kernel<<<blocks, threads>>>(
		derivatives.data(), grad_input.data(), N, C, H, W, scale
	);

	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	return grad_input;
}

}