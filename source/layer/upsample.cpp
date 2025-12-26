#include "layer.hpp"
#include <algorithm>
#include <omp.h>

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

	const float* in_ptr = input.data();
	float* out_ptr = output.data();

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			const float* img_in = in_ptr + (n * C + c) * H * W;
			float* img_out = out_ptr + (n * C + c) * out_h * out_w;

			// Iterate over input rows
			for(int ih = 0; ih < H; ++ih) {
				const float* in_row = img_in + ih * W;
				
				// For each input row, fill 'scale' number of output rows
				for(int sh = 0; sh < scale; ++sh) {
					float* out_row = img_out + (ih * scale + sh) * out_w;

					// Inner loop over input width
					// Compiler can vectorize this copy-scatter pattern
					for(int iw = 0; iw < W; ++iw) {
						float val = in_row[iw];
						// Unroll the small scale factor filling
						for(int sw = 0; sw < scale; ++sw) {
							out_row[iw * scale + sw] = val;
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
Tensor<Device::CPU> UpSample2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	const int N = cached_input.batches();
	const int C = cached_input.channels();
	const int H = cached_input.height();
	const int W = cached_input.width();
	const int out_h = grad_output.height();
	const int out_w = grad_output.width();

	Tensor<Device::CPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);
	
	const int total_size = derivatives.size();
	const float* go_ptr = grad_output.data();
	float* der_ptr = derivatives.data();

	// Vectorized element-wise multiplication
	#pragma omp parallel for simd
	for(int i = 0; i < total_size; ++i) {
		derivatives.data()[i] *= grad_output.data()[i];
	}

	Tensor<Device::CPU> grad_input(N, C, H, W);
	grad_input.fill(0.0f);
	
	float* gi_ptr = grad_input.data();

	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int c = 0; c < C; ++c) {
			float* cur_gi = gi_ptr + (n * C + c) * H * W;
			const float* cur_der = der_ptr + (n * C + c) * out_h * out_w;

			// Iterate Output in blocks corresponding to Input pixels
			for(int ih = 0; ih < H; ++ih) {
				for(int iw = 0; iw < W; ++iw) {
					
					// Sum up the 'scale x scale' block from derivative
					float sum = 0.0f;
					int oh_start = ih * scale;
					int ow_start = iw * scale;

					for(int sh = 0; sh < scale; ++sh) {
						const float* der_row = cur_der + (oh_start + sh) * out_w;
						for(int sw = 0; sw < scale; ++sw) {
							sum += der_row[ow_start + sw];
						}
					}
					cur_gi[ih * W + iw] = sum;
				}
			}
		}
	}
	return grad_input;
}

}