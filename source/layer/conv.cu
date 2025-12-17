#include "layer.hpp"
#include "utils/error.cuh"

// Convolutional 2D GPU Kernels

__global__ void conv2d_forward_kernel(
	const float *input, const float *weights, const float *biases, float *output,
	int batch, int in_channels, int out_channels,
	int input_h, int input_w,
	int kernel_size, int stride, int padding,
	int output_h, int output_w)
{
	int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_output = batch * out_channels * output_h * output_w;

	if(output_idx >= total_output) return;

	int n = output_idx / (out_channels * output_h * output_w);
	int remainder = output_idx % (out_channels * output_h * output_w);
	int oc = remainder / (output_h * output_w);
	remainder = remainder % (output_h * output_w);
	int oh = remainder / output_w;
	int ow = remainder % output_w;

	float sum = biases[oc];

	for(int ic = 0; ic < in_channels; ++ic) {
		for(int kh = 0; kh < kernel_size; ++kh) {
			for(int kw = 0; kw < kernel_size; ++kw) {
				int ih = oh * stride + kh - padding;
				int iw = ow * stride + kw - padding;

				if(ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) continue;

				int input_idx = ((n * in_channels + ic) * input_h + ih) * input_w + iw;
				int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

				sum += input[input_idx] * weights[weight_idx];
			}
		}
	}

	output[output_idx] = sum;
}

__global__ void conv2d_bias_grad_kernel(const float *grad_out, float *grad_b, int N, int OC, int H_out, int W_out) {
	int oc = blockIdx.x * blockDim.x + threadIdx.x;
	if(oc >= OC) return;
	
	float sum = 0.0f;
	for(int n = 0; n < N; ++n)
		for(int oh = 0; oh < H_out; ++oh)
			for(int ow = 0; ow < W_out; ++ow)
				sum += grad_out[((n * OC + oc) * H_out + oh) * W_out + ow];
	
	grad_b[oc] = sum;
}

__global__ void conv2d_weight_grad_kernel(
	const float *input, const float *grad_out, float *grad_w,
	int N, int IC, int OC, int H_in, int W_in,
	int K, int stride, int pad, int H_out, int W_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = OC * IC * K * K;
	if(idx >= total) return;
	
	int oc = idx / (IC * K * K);
	int rem = idx % (IC * K * K);
	int ic = rem / (K * K);
	rem = rem % (K * K);
	int kh = rem / K;
	int kw = rem % K;

	float sum = 0.0f;
	for(int n = 0; n < N; ++n) {
		for(int oh = 0; oh < H_out; ++oh) {
			for(int ow = 0; ow < W_out; ++ow) {
				int ih = oh * stride + kh - pad;
				int iw = ow * stride + kw - pad;
				if(ih < 0 || ih >= H_in || iw < 0 || iw >= W_in) continue;
				
				int in_idx = ((n * IC + ic) * H_in + ih) * W_in + iw;
				int go_idx = ((n * OC + oc) * H_out + oh) * W_out + ow;
				sum += input[in_idx] * grad_out[go_idx];
			}
		}
	}
	grad_w[idx] = sum;
}

__global__ void conv2d_input_grad_kernel(
	const float *grad_out, const float *weights, float *grad_in,
	int N, int IC, int OC, int H_in, int W_in,
	int K, int stride, int pad, int H_out, int W_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * IC * H_in * W_in;
	if(idx >= total) return;
	
	int n = idx / (IC * H_in * W_in);
	int rem = idx % (IC * H_in * W_in);
	int ic = rem / (H_in * W_in);
	rem = rem % (H_in * W_in);
	int ih = rem / W_in;
	int iw = rem % W_in;

	float sum = 0.0f;
	for(int oc = 0; oc < OC; ++oc) {
		for(int kh = 0; kh < K; ++kh) {
			for(int kw = 0; kw < K; ++kw) {
				int oh_unstrided = ih + pad - kh;
				int ow_unstrided = iw + pad - kw;
				
				if(oh_unstrided < 0 || ow_unstrided < 0) continue;
				if(oh_unstrided % stride != 0 || ow_unstrided % stride != 0) continue;
				
				int oh = oh_unstrided / stride;
				int ow = ow_unstrided / stride;
				
				if(oh < 0 || oh >= H_out || ow < 0 || ow >= W_out) continue;

				int go_idx = ((n * OC + oc) * H_out + oh) * W_out + ow;
				int w_idx = ((oc * IC + ic) * K + kh) * K + kw;
				sum += grad_out[go_idx] * weights[w_idx];
			}
		}
	}
	grad_in[idx] = sum;
}

// Convolutional 2D GPU specialization implementations

namespace Layer {

template <>
Tensor<Device::GPU> Conv2D<Device::GPU>::forward(const Tensor<Device::GPU> &input) {
	this->cached_input = input;

	int N = input.batches();
	int H = input.height();
	int W = input.width();

	int out_h = (H + 2 * padding - filter_size) / stride + 1;
	int out_w = (W + 2 * padding - filter_size) / stride + 1;

	Tensor<Device::GPU> output(N, out_channels, out_h, out_w);
	size_t out_size = output.size();

	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	int blocks = (out_size + threads - 1) / threads;

	conv2d_forward_kernel<<<blocks, threads>>>(
		input.data(), weights.data(), biases.data(), output.data(),
		N, in_channels, out_channels, H, W,
		filter_size, stride, padding, out_h, out_w
	);

	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	return output;
}

template <>
Tensor<Device::GPU> Conv2D<Device::GPU>::backward(const Tensor<Device::GPU> &grad_output) {
	int N = cached_input.batches();
	int C = in_channels;
	int H = cached_input.height();
	int W = cached_input.width();
	
	int out_h = grad_output.height();
	int out_w = grad_output.width();

	// Initialize gradients to 0
	Tensor<Device::GPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);
	grad_weights.fill(0.0f);
	grad_biases.fill(0.0f);

	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	
	// Bias Gradients
	int blocks_b = (out_channels + threads - 1) / threads;
	conv2d_bias_grad_kernel<<<blocks_b, threads>>>(
		grad_output.data(), grad_biases.data(),
		N, out_channels, out_h, out_w
	);
	checkCUDA(cudaGetLastError());

	// Weight Gradients
	size_t w_size = grad_weights.size();
	int blocks_w = (w_size + threads - 1) / threads;
	conv2d_weight_grad_kernel<<<blocks_w, threads>>>(
		cached_input.data(), grad_output.data(), grad_weights.data(),
		N, C, out_channels, H, W,
		filter_size, stride, padding, out_h, out_w
	);
	checkCUDA(cudaGetLastError());

	// Input Gradients
	size_t in_size = grad_input.size();
	int blocks_in = (in_size + threads - 1) / threads;
	conv2d_input_grad_kernel<<<blocks_in, threads>>>(
		grad_output.data(), weights.data(), grad_input.data(),
		N, C, out_channels, H, W,
		filter_size, stride, padding, out_h, out_w
	);
	
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());

	std::visit([&](auto&& act) { backward_activate(grad_input, act); }, this->activation);
	return grad_input;
}

}