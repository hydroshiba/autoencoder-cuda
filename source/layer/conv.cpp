#include "layer.hpp"
#include <algorithm>

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

	return output;
}

template <>
Tensor<Device::CPU> Conv2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	const int N = cached_input.batches();
	const int H = cached_input.height();
	const int W = cached_input.width();

	const int out_h = grad_output.height();
	const int out_w = grad_output.width();

	Tensor<Device::CPU> grad_input(N, in_channels, H, W);
	
	// Zero initialize gradients
	grad_input.fill(0.0f);
	grad_weights.fill(0.0f);
	grad_biases.fill(0.0f);

	// Bias gradient
	for(int n = 0; n < N; ++n) {
		for(int oc = 0; oc < out_channels; ++oc) {
			float bsum = 0.0f;
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					bsum += grad_output.data()[go_idx];
				}
			}
			grad_biases.data()[oc] += bsum;
		}
	}

	// Weight and Input gradients
	for(int n = 0; n < N; ++n) {
		for(int oc = 0; oc < out_channels; ++oc) {
			for(int oh = 0; oh < out_h; ++oh) {
				for(int ow = 0; ow < out_w; ++ow) {
					
					int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
					float go = grad_output.data()[go_idx];

					for(int ic = 0; ic < in_channels; ++ic) {
						for(int kh = 0; kh < filter_size; ++kh) {
							for(int kw = 0; kw < filter_size; ++kw) {
								
								int ih = oh * stride + kh - padding;
								int iw = ow * stride + kw - padding;

								if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
									int in_idx = ((n * in_channels + ic) * H + ih) * W + iw;
									int w_idx = ((oc * in_channels + ic) * filter_size + kh) * filter_size + kw;

									// dW += input * grad_output
									grad_weights.data()[w_idx] += cached_input.data()[in_idx] * go;

									// dInput += weight * grad_output
									grad_input.data()[in_idx] += weights.data()[w_idx] * go;
								}
							}
						}
					}
				}
			}
		}
	}

	return grad_input;
}

}