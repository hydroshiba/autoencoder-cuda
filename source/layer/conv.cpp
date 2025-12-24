#include "layer.hpp"
#include <algorithm>
#include <omp.h>

// Convolutional 2D CPU specialization implementations

namespace Layer {

template <>
Tensor<Device::CPU> Conv2D<Device::CPU>::forward(const Tensor<Device::CPU> &input) {
	this->cached_input = input;

	const int N = input.batches();
	const int H = input.height();
	const int W = input.width();

	const int h_num = H + 2 * padding - filter_size;
	const int w_num = W + 2 * padding - filter_size;
	
	const int out_h = h_num / stride + 1;
	const int out_w = w_num / stride + 1;

	Tensor<Device::CPU> output(N, out_channels, out_h, out_w);

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int oc = 0; oc < out_channels; ++oc) {
			float bias = biases.data()[oc];
			
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					float sum = bias;

					for(int ic = 0; ic < in_channels; ++ic) {
						for(int kh = 0; kh < filter_size; ++kh) {
							for(int kw = 0; kw < filter_size; ++kw) {
								
								int ih = oh * stride + kh - padding;
								int iw = ow * stride + kw - padding;

								if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
									int in_idx = ((n * in_channels + ic) * H + ih) * W + iw;
									int w_idx = ((oc * in_channels + ic) * filter_size + kh) * filter_size + kw;
									sum += input.data()[in_idx] * weights.data()[w_idx];
								}
							}
						}
					}
					
					int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					output.data()[out_idx] = sum;
				}
			}
		}
	}

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	this->cached_output = output;
	return output;
}

template <>
Tensor<Device::CPU> Conv2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	Tensor<Device::CPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);

	#pragma omp parallel for
	for(size_t i = 0; i < grad_output.size(); ++i)
		derivatives.data()[i] *= grad_output.data()[i];

	const int N = cached_input.batches();
	const int H = cached_input.height();
	const int W = cached_input.width();
	const int out_h = derivatives.height();
	const int out_w = derivatives.width();

	Tensor<Device::CPU> grad_input(N, in_channels, H, W);
	grad_input.fill(0.0f);

	// 1. Bias Gradients (Fast enough to run serially or parallelize over OC)
	#pragma omp parallel for
	for(int oc = 0; oc < out_channels; ++oc) {
		float bsum = 0.0f;
		for(int n = 0; n < N; ++n) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					bsum += derivatives.data()[go_idx];
				}
			}
		}
		grad_biases.data()[oc] += bsum;
	}

	// 2. Input Gradients (Parallelize over BATCH 'n')
	// Safe because every thread writes to a distinct slice of 'grad_input'
	#pragma omp parallel for
	for(int n = 0; n < N; ++n) {
		for(int oc = 0; oc < out_channels; ++oc) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					float go = derivatives.data()[go_idx];

					for(int ic = 0; ic < in_channels; ++ic) {
						for(int kh = 0; kh < filter_size; ++kh) {
							for(int kw = 0; kw < filter_size; ++kw) {
								int ih = oh * stride + kh - padding;
								int iw = ow * stride + kw - padding;

								if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
									int in_idx = ((n * in_channels + ic) * H + ih) * W + iw;
									int w_idx = ((oc * in_channels + ic) * filter_size + kh) * filter_size + kw;

									// CALCULATE INPUT GRAD ONLY
									grad_input.data()[in_idx] += weights.data()[w_idx] * go;
								}
							}
						}
					}
				}
			}
		}
	}

	// 3. Weight Gradients (Parallelize over OUTPUT CHANNEL 'oc')
	// Safe because every thread writes to a distinct filter in 'grad_weights'
	#pragma omp parallel for
	for(int oc = 0; oc < out_channels; ++oc) {
		for(int n = 0; n < N; ++n) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					float go = derivatives.data()[go_idx];

					for(int ic = 0; ic < in_channels; ++ic) {
						for(int kh = 0; kh < filter_size; ++kh) {
							for(int kw = 0; kw < filter_size; ++kw) {
								int ih = oh * stride + kh - padding;
								int iw = ow * stride + kw - padding;

								if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
									int in_idx = ((n * in_channels + ic) * H + ih) * W + iw;
									int w_idx = ((oc * in_channels + ic) * filter_size + kh) * filter_size + kw;
									grad_weights.data()[w_idx] += cached_input.data()[in_idx] * go;
								}
							}
						}
					}
				}
			}
		}
	}

	// std::visit([&](auto&& act) { backward_activate(grad_input, act); }, this->activation);
	return grad_input;
}

}