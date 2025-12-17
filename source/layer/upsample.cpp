#include "layer.hpp"
#include <algorithm>

// Upsample 2D CPU specialization implementations

namespace Layer {

template <>
Tensor<Device::CPU> UpSample2D<Device::CPU>::forward(const Tensor<Device::CPU> &input) {
	this->cached_input = input;

	const int N = input.batches();
	const int C = input.channels();
	const int H = input.height();
	const int W = input.width();

	const int out_h = H * scale;
	const int out_w = W * scale;

	Tensor<Device::CPU> output(N, C, out_h, out_w);

	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			for(int h = 0; h < out_h; ++h) {
				for(int w = 0; w < out_w; ++w) {
					int ih = h / scale;
					int iw = w / scale;

					int in_idx = ((n * C + c) * H + ih) * W + iw;
					int out_idx = ((n * C + c) * out_h + h) * out_w + w;

					output.data()[out_idx] = input.data()[in_idx];
				}
			}
		}
	}

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	return output;
}

template <>
Tensor<Device::CPU> UpSample2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	const int N = cached_input.batches();
	const int C = cached_input.channels();
	const int H = cached_input.height();
	const int W = cached_input.width();

	Tensor<Device::CPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);

	const int out_h = H * scale;
	const int out_w = W * scale;

	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			for(int h = 0; h < out_h; ++h) {
				for(int w = 0; w < out_w; ++w) {
					int ih = h / scale;
					int iw = w / scale;

					int in_idx = ((n * C + c) * H + ih) * W + iw;
					int out_idx = ((n * C + c) * out_h + h) * out_w + w;

					grad_input.data()[in_idx] += grad_output.data()[out_idx];
				}
			}
		}
	}

	std::visit([&](auto&& act) { backward_activate(grad_input, act); }, this->activation);
	return grad_input;
}

}