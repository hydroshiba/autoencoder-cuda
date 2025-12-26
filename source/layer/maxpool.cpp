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

	const float* in_base = input.data();
	float* out_base = output.data();
	float* mask_base = mask.data();

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			const float* in_slice = in_base + (n * C + c) * H * W;
			float* out_slice = out_base + (n * C + c) * out_h * out_w;
			float* mask_slice = mask_base + (n * C + c) * out_h * out_w;

			// Initialize with lowest possible value
			float init_val = -std::numeric_limits<float>::infinity();
			for(int i = 0; i < out_h * out_w; ++i) out_slice[i] = init_val;

			// LOOP INVERSION: Iterate Kernel Outer, Image Inner
			// This allows vectorizing the updates over the image rows
			for(int kh = 0; kh < pool_size; ++kh) {
				for(int kw = 0; kw < pool_size; ++kw) {
					int local_idx = kh * pool_size + kw;

					for(int oh = 0; oh < out_h; ++oh) {
						int ih = oh * pool_size + kh;
						const float* in_row = in_slice + ih * W;
						float* out_row = out_slice + oh * out_w;
						float* mask_row = mask_slice + oh * out_w;

						int w_base_offset = kw; // iw = ow * pool_size + kw

						// Vectorized Max Update
						#pragma omp simd
						for(int ow = 0; ow < out_w; ++ow) {
							int iw = ow * pool_size + w_base_offset;
							float val = in_row[iw];
							
							// Conditional update (supported by AVX)
							if (val > out_row[ow]) {
								out_row[ow] = val;
								mask_row[ow] = static_cast<float>(local_idx);
							}
						}
					}
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
	
	#pragma omp parallel for simd
	for(size_t i = 0; i < derivatives.size(); ++i) {
		derivatives.data()[i] *= grad_output.data()[i];
	}

	Tensor<Device::CPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);

	float* gi_base = grad_input.data();
	const float* der_base = derivatives.data();
	const float* mask_base = mask.data();

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			float* gi_slice = gi_base + (n * C + c) * H * W;
			const float* der_slice = der_base + (n * C + c) * out_h * out_w;
			const float* mask_slice = mask_base + (n * C + c) * out_h * out_w;

			// We can't easily invert loop here because we need to scatter based on mask.
			// However, we can keep memory access linear on Deriv/Mask side.
			for(int i = 0; i < out_h * out_w; ++i) {
				int local_idx = static_cast<int>(mask_slice[i]);
				int kh = local_idx / pool_size;
				int kw = local_idx % pool_size;
				
				int oh = i / out_w;
				int ow = i % out_w;

				int ih = oh * pool_size + kh;
				int iw = ow * pool_size + kw;

				gi_slice[ih * W + iw] += der_slice[i];
			}
		}
	}
	return grad_input;
}

}