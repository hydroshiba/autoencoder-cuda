#include "layer.hpp"
#include <algorithm>
#include <limits>
#include <omp.h>

// Max Pooling 2D CPU specialization implementations

namespace Layer {

template <>
Tensor<Device::CPU> MaxPool2D<Device::CPU>::forward(const Tensor<Device::CPU> &input) {
	this->cached_input = input;

	const int N = input.batches();
	const int C = input.channels();
	const int H = input.height();
	const int W = input.width();

	const int out_h = H / pool_size;
	const int out_w = W / pool_size;

	Tensor<Device::CPU> output(N, C, out_h, out_w);
	this->mask = Tensor<Device::CPU>(N, C, out_h, out_w);

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					float max_val = -std::numeric_limits<float>::infinity();
					int max_local = 0;

					int h_start = oh * pool_size;
					int w_start = ow * pool_size;

					for(int kh = 0; kh < pool_size; ++kh) {
						for(int kw = 0; kw < pool_size; ++kw) {
							
							int ih = h_start + kh;
							int iw = w_start + kw;

							int idx = ((n * C + c) * H + ih) * W + iw;
							float v = input.data()[idx];

							if(v > max_val) {
								max_val = v;
								// Store relative local index
								max_local = kh * pool_size + kw;
							}
						}
					}

					int out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
					output.data()[out_idx] = max_val;
					mask.data()[out_idx] = static_cast<float>(max_local);
				}
			}
		}
	}

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	this->cached_output = output;
	return output;
}

template <>
Tensor<Device::CPU> MaxPool2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	const int N = cached_input.batches();
	const int C = cached_input.channels();
	const int H = cached_input.height();
	const int W = cached_input.width();
	
	const int out_h = grad_output.height();
	const int out_w = grad_output.width();

	Tensor<Device::CPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);
	for(size_t i = 0; i < derivatives.size(); ++i) {
		derivatives.data()[i] *= grad_output.data()[i];
	}

	Tensor<Device::CPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);

	// Parallelize over outputs, safe to write to input because mapping is unique
	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					int out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
					
					// Retrieve local index
					int local_idx = static_cast<int>(mask.data()[out_idx]);
					int kh = local_idx / pool_size;
					int kw = local_idx % pool_size;

					int ih = oh * pool_size + kh;
					int iw = ow * pool_size + kw;
					
					int in_idx = ((n * C + c) * H + ih) * W + iw;
					grad_input.data()[in_idx] += derivatives.data()[out_idx];
				}
			}
		}
	}
	return grad_input;
}

}