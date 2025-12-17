#include "layer.hpp"
#include <algorithm>
#include <limits>

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

	int n, c, oh, ow, kh, kw;

	for(n = 0; n < N; ++n) {
		for(c = 0; c < C; ++c) {
			for(oh = 0; oh < out_h; ++oh) {
				for(ow = 0; ow < out_w; ++ow) {
					
					float max_val = -std::numeric_limits<float>::infinity();
					int max_idx = -1;

					// Base index for input window
					int h_start = oh * pool_size;
					int w_start = ow * pool_size;

					for(kh = 0; kh < pool_size; ++kh) {
						for(kw = 0; kw < pool_size; ++kw) {
							
							int ih = h_start + kh;
							int iw = w_start + kw;

							// Manual flat index calculation to bypass operator() bounds check
							int idx = ((n * C + c) * H + ih) * W + iw;
							float v = input.data()[idx];

							if(v > max_val) {
								max_val = v;
								max_idx = idx;
							}
						}
					}

					int out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
					output.data()[out_idx] = max_val;
					// Store index as float in mask tensor
					mask.data()[out_idx] = static_cast<float>(max_idx);
				}
			}
		}
	}

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

	Tensor<Device::CPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);

	int total_output_elements = N * C * out_h * out_w;

	for(int i = 0; i < total_output_elements; ++i) {
		int in_idx = static_cast<int>(mask.data()[i]);
		grad_input.data()[in_idx] += grad_output.data()[i];
	}

	return grad_input;
}

}