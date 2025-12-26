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
	
	const float* in_base = input.data();
	const float* w_base = weights.data();
	const float* b_base = biases.data();
	float* out_base = output.data();

	// Parallelize over Batch and Output Channel (Independent units)
	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int oc = 0; oc < out_channels; ++oc) {
			
			float bias = b_base[oc];
			float* out_slice = out_base + (n * out_channels + oc) * out_h * out_w;
			const float* in_batch = in_base + n * in_channels * H * W;

			// Initialize output with bias (Vectorized fill)
			for(int i = 0; i < out_h * out_w; ++i) out_slice[i] = bias;

			// Loop Order: OH -> IC -> KH -> KW -> OW
			// Putting OW inner-most allows vectorization (contiguous writes)
			for(int oh = 0; oh < out_h; ++oh) {
				float* out_row = out_slice + oh * out_w;
				
				// Pre-calc Vertical Input Bounds
				int h_offset = oh * stride - padding;
				int kh_start = std::max(0, -h_offset);
				int kh_end = std::min(filter_size, H - h_offset);

				for(int ic = 0; ic < in_channels; ++ic) {
					const float* w_slice = w_base + (oc * in_channels + ic) * filter_size * filter_size;
					const float* in_slice = in_batch + ic * H * W;

					for(int kh = kh_start; kh < kh_end; ++kh) {
						int ih = h_offset + kh;
						const float* in_row = in_slice + ih * W;
						
						for(int kw = 0; kw < filter_size; ++kw) {
							// Weight is scalar in the inner-most loop!
							float w_val = w_slice[kh * filter_size + kw];
							
							// Pre-calc Horizontal Input Bounds for Vectorization
							int w_base_offset = -padding + kw;
							
							// Calculate loop bounds for ow to ensure input access is valid
							// iw = ow * stride + w_base_offset
							// 0 <= iw < W
							int ow_start = 0;
							int ow_end = out_w;

							// Refine bounds based on padding logic to remove IF inside loop
							if (stride == 1) {
								ow_start = std::max(0, -w_base_offset);
								ow_end = std::min(out_w, W - w_base_offset);
							} else {
								// For stride > 1, simple clamping is harder, fall back to check or careful math
								// Keeping simple check for stride > 1 cases, but optimizing stride=1 (90% cases)
							}

							if (stride == 1) {
								// HOT PATH: Stride 1 (Vectorized FMA)
								#pragma omp simd
								for(int ow = ow_start; ow < ow_end; ++ow) {
									out_row[ow] += w_val * in_row[ow + w_base_offset];
								}
							} else {
								// Stride > 1 Path
								#pragma omp simd
								for(int ow = 0; ow < out_w; ++ow) {
									int iw = ow * stride + w_base_offset;
									if(iw >= 0 && iw < W) {
										out_row[ow] += w_val * in_row[iw];
									}
								}
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
Tensor<Device::CPU> Conv2D<Device::CPU>::backward(const Tensor<Device::CPU> &grad_output) {
	Tensor<Device::CPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);

	#pragma omp parallel for simd
	for(size_t i = 0; i < grad_output.size(); ++i)
		derivatives.data()[i] *= grad_output.data()[i];

	const int N = cached_input.batches();
	const int H = cached_input.height();
	const int W = cached_input.width();
	const int out_h = derivatives.height();
	const int out_w = derivatives.width();

	Tensor<Device::CPU> grad_input(N, in_channels, H, W);
	grad_input.fill(0.0f);

	// 1. Bias Gradients
	#pragma omp parallel for
	for(int oc = 0; oc < out_channels; ++oc) {
		float bsum = 0.0f;
		for(int n = 0; n < N; ++n) {
			const float* der_ptr = derivatives.data() + (n * out_channels + oc) * out_h * out_w;
			// Vector reduction
			#pragma omp simd reduction(+:bsum)
			for(int i = 0; i < out_h * out_w; ++i) bsum += der_ptr[i];
		}
		grad_biases.data()[oc] += bsum;
	}

	// 2. Input Gradients
	// Loop order optimized for accumulating into GradInput row
	#pragma omp parallel for collapse(2)
	for(int n = 0; n < N; ++n) {
		for(int ic = 0; ic < in_channels; ++ic) {
			float* gi_slice = grad_input.data() + (n * in_channels + ic) * H * W;
			
			for(int oc = 0; oc < out_channels; ++oc) {
				const float* w_slice = weights.data() + (oc * in_channels + ic) * filter_size * filter_size;
				const float* der_slice = derivatives.data() + (n * out_channels + oc) * out_h * out_w;

				for(int oh = 0; oh < out_h; ++oh) {
					const float* der_row = der_slice + oh * out_w;
					
					int h_offset = oh * stride - padding;
					int kh_start = std::max(0, -h_offset);
					int kh_end = std::min(filter_size, H - h_offset);
					
					for(int kh = kh_start; kh < kh_end; ++kh) {
						int ih = h_offset + kh;
						float* gi_row = gi_slice + ih * W;

						for(int kw = 0; kw < filter_size; ++kw) {
							float w_val = w_slice[kh * filter_size + kw];
							int w_offset = -padding + kw; // Base offset

							// Accumulate Deriv * Weight into Input Grad
							// Similar vectorization logic as Forward
							if (stride == 1) {
								int ow_start = std::max(0, -w_offset);
								int ow_end = std::min(out_w, W - w_offset); // Clamp to W relative to input
								
								// Wait, strict bounds: 
								// iw = ow + w_offset. We need 0 <= iw < W.
								// ow >= -w_offset. ow < W - w_offset.
								// AND we need 0 <= ow < out_w.
								
								int valid_ow_start = std::max(0, std::max(0, -w_offset));
								int valid_ow_end = std::min(out_w, W - w_offset);

								#pragma omp simd
								for(int ow = valid_ow_start; ow < valid_ow_end; ++ow) {
									gi_row[ow + w_offset] += w_val * der_row[ow];
								}
							} else {
								#pragma omp simd
								for(int ow = 0; ow < out_w; ++ow) {
									int iw = ow * stride + w_offset;
									if (iw >= 0 && iw < W) {
										gi_row[iw] += w_val * der_row[ow];
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// 3. Weight Gradients
	// Optimized: Inner loop over Output Width (Vector Dot Product)
	#pragma omp parallel for collapse(2)
	for(int oc = 0; oc < out_channels; ++oc) {
		for(int ic = 0; ic < in_channels; ++ic) {
			float* gw_slice = grad_weights.data() + (oc * in_channels + ic) * filter_size * filter_size;

			for(int n = 0; n < N; ++n) {
				const float* in_slice = cached_input.data() + (n * in_channels + ic) * H * W;
				const float* der_slice = derivatives.data() + (n * out_channels + oc) * out_h * out_w;

				for(int kh = 0; kh < filter_size; ++kh) {
					for(int kw = 0; kw < filter_size; ++kw) {
						
						// Accumulate sum for this specific weight scalar
						float sum = 0.0f;
						int w_offset = -padding + kw;

						for(int oh = 0; oh < out_h; ++oh) {
							const float* der_row = der_slice + oh * out_w;
							
							int ih = oh * stride - padding + kh;
							if(ih < 0 || ih >= H) continue;

							const float* in_row = in_slice + ih * W;

							if(stride == 1) {
								int valid_ow_start = std::max(0, -w_offset);
								int valid_ow_end = std::min(out_w, W - w_offset);
								
								// Vectorized Dot Product
								#pragma omp simd reduction(+:sum)
								for(int ow = valid_ow_start; ow < valid_ow_end; ++ow) {
									sum += in_row[ow + w_offset] * der_row[ow];
								}
							} else {
								#pragma omp simd reduction(+:sum)
								for(int ow = 0; ow < out_w; ++ow) {
									int iw = ow * stride + w_offset;
									if(iw >= 0 && iw < W) {
										sum += in_row[iw] * der_row[ow];
									}
								}
							}
						}
						gw_slice[kh * filter_size + kw] += sum;
					}
				}
			}
		}
	}
	return grad_input;
}

}