#include "layer.hpp"
#include "utils/error.cuh"
#include "utils/kernel.cuh"

// --------------------------------------------------------------------------
// Helper Kernels
// --------------------------------------------------------------------------

__global__ void conv2d_im2col(
	const float* data_im, float* data_col,
	int batch_size, int channels, int height, int width,
	int ksize, int pad, int stride,
	int height_col, int width_col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int spatial_area = height_col * width_col;
	int total_cols = batch_size * spatial_area;

	if(idx >= total_cols) return;

	int w_out = idx % width_col;
	int h_out = (idx / width_col) % height_col;
	int b = idx / spatial_area;

	int input_offset = b * (channels * height * width);
	int matrix_col_idx = idx;

	for(int c = 0; c < channels; ++c) {
		for(int kh = 0; kh < ksize; ++kh) {
			for(int kw = 0; kw < ksize; ++kw) {
				int h_in = h_out * stride - pad + kh;
				int w_in = w_out * stride - pad + kw;
				int matrix_row_idx = (c * ksize + kh) * ksize + kw;
				int dst_idx = matrix_row_idx * total_cols + matrix_col_idx;

				float val = 0.0f;
				if(h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
					val = data_im[input_offset + (c * height + h_in) * width + w_in];
				}
				data_col[dst_idx] = val;
			}
		}
	}
}

__global__ void permute_output_add_bias(
	const float* gemm_out, const float* bias, float* final_out,
	int batch, int channels, int height, int width)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int spatial = height * width;
	int total = batch * channels * spatial;

	if(idx >= total) return;

	int w_out = idx % width;
	int rem = idx / width;
	int h_out = rem % height;
	rem /= height;
	int c = rem % channels;
	int b = rem / channels;

	int src_flat_idx = b * spatial + h_out * width + w_out; 
	int src_idx = c * (batch * spatial) + src_flat_idx;

	final_out[idx] = gemm_out[src_idx] + bias[c];
}

// Permutes (Batch, Channel, Height, Width) -> (Channel, Batch, Height, Width)
__global__ void permute_nchw_to_cnhw(
	const float* input, float* output,
	int N, int C, int Spatial) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * C * Spatial;
	if(idx >= total) return;

	int s = idx % Spatial;
	int temp = idx / Spatial;
	int c = temp % C;
	int n = temp / C;

	int out_idx = c * (N * Spatial) + n * Spatial + s;
	output[out_idx] = input[idx];
}

// Simple reduction to sum rows of the (C x N*H*W) matrix for bias gradients
__global__ void reduce_sum_rows(
	const float* input, float* biases,
	int rows, int cols)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row >= rows) return;

	float sum = 0.0f;
	for(int i = 0; i < cols; ++i) {
		sum += input[row * cols + i];
	}
	biases[row] = sum;
}

static __global__ void conv2d_col2im(
	const float *data_col, float *data_im,
	int N, int C, int H, int W,
	int K, int stride, int pad,
	int out_h, int out_w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_kernels = N * C * H * W;
	
	if(idx >= num_kernels) return;

	int w_in = idx % W;
	int temp = idx / W;
	int h_in = temp % H;
	temp /= H;
	int c = temp % C;
	int n = temp / C;

	float val = 0.0f;
	int total_cols = N * out_h * out_w;

	for(int kh = 0; kh < K; ++kh) {
		int h_val = h_in + pad - kh;
		if(h_val >= 0 && (h_val % stride == 0)) {
			int h_out = h_val / stride;
			if(h_out < out_h) {
				for(int kw = 0; kw < K; ++kw) {
					int w_val = w_in + pad - kw;
					if(w_val >= 0 && (w_val % stride == 0)) {
						int w_out = w_val / stride;
						if(w_out < out_w) {
							int matrix_row = (c * K + kh) * K + kw;
							int matrix_col = n * (out_h * out_w) + h_out * out_w + w_out;
							val += data_col[matrix_row * total_cols + matrix_col];
						}
					}
				}
			}
		}
	}
	data_im[idx] = val;
}

// --------------------------------------------------------------------------
// Conv2D Implementation
// --------------------------------------------------------------------------

namespace Layer {

template <>
Tensor<Device::GPU> Conv2D<Device::GPU>::forward(const Tensor<Device::GPU> &input) {
	this->cached_input = input;

	int N = input.batches();
	int H = input.height();
	int W = input.width();

	int out_h = (H + 2 * padding - filter_size) / stride + 1;
	int out_w = (W + 2 * padding - filter_size) / stride + 1;

	int m = out_channels;
	int k = in_channels * filter_size * filter_size;
	int n_cols = N * out_h * out_w; 

	// ----------------------------------------------------------------------
	// Forward Buffer Allocation
	// ----------------------------------------------------------------------
	if (!forward_buffer) {
		constexpr int MAX_BATCH_FWD = 256;
		
		size_t pixels_max = (size_t)out_h * out_w;
		size_t k_dim = (size_t)in_channels * filter_size * filter_size;
		size_t m_dim = (size_t)out_channels;
		size_t cols_fwd = (size_t)MAX_BATCH_FWD * pixels_max;

		checkCUDA(cudaMalloc(&forward_buffer, k_dim * cols_fwd * sizeof(float)));
		checkCUDA(cudaMalloc(&forward_gemm_proxy, m_dim * cols_fwd * sizeof(float)));
	}

	Tensor<Device::GPU> output(N, out_channels, out_h, out_w);
	dim3 threads(Kernel::BLOCK_SIZE, Kernel::BLOCK_SIZE);

	// 1. Batched Im2Col
	int total_pixels = n_cols;
	dim3 grid_im2col((total_pixels + 255) / 256);
	conv2d_im2col<<<grid_im2col, 256>>>(
		input.data(), forward_buffer,
		N, in_channels, H, W,
		filter_size, padding, stride,
		out_h, out_w
	);

	// 2. GEMM
	dim3 grid_gemm(
		(n_cols + threads.x - 1) / threads.x,
		(m + threads.y - 1) / threads.y
	);
	Kernel::matrix_multiply<<<grid_gemm, threads>>>(
		weights.data(), forward_buffer, forward_gemm_proxy,
		m, n_cols, k
	);

	// 3. Permute + Bias
	int out_size = output.size();
	int blocks_perm = (out_size + 255) / 256;
	permute_output_add_bias<<<blocks_perm, 256>>>(
		forward_gemm_proxy, biases.data(), output.data(),
		N, out_channels, out_h, out_w
	);

	std::visit([&](auto&& act) { forward_activate(output, act); }, this->activation);
	this->cached_output = output;
	return output;
}

template <>
Tensor<Device::GPU> Conv2D<Device::GPU>::backward(const Tensor<Device::GPU> &grad_output) {
	Tensor<Device::GPU> derivatives = cached_output;
	std::visit([&](auto&& act) { backward_activate(derivatives, act); }, this->activation);

	int threads = Config::Conv2D::block_width * Config::Conv2D::block_height;
	int blocks_deriv = (derivatives.size() + threads - 1) / threads;
	Kernel::vector_multiply<<<blocks_deriv, threads>>>(
		derivatives.data(), grad_output.data(), derivatives.size()
	);

	int N = cached_input.batches();
	int C_in = in_channels;
	int H_in = cached_input.height();
	int W_in = cached_input.width();
	
	int C_out = out_channels;
	int H_out = derivatives.height();
	int W_out = derivatives.width();

	int K_sz = C_in * filter_size * filter_size; 
	int L_sz = N * H_out * W_out; 
	int M_sz = C_out; 

	// ----------------------------------------------------------------------
	// Backward Buffer Allocation
	// ----------------------------------------------------------------------
	if (!backward_dY_permute) {
		constexpr int MAX_BATCH_BWD = 64;

		// Use dimensions from the derivatives (grad_output) to determine spatial size
		size_t pixels_max = (size_t)H_out * W_out;
		
		size_t k_dim = (size_t)K_sz;
		size_t m_dim = (size_t)M_sz;
		size_t cols_bwd = (size_t)MAX_BATCH_BWD * pixels_max;

		checkCUDA(cudaMalloc(&backward_dY_permute, m_dim * cols_bwd * sizeof(float)));
		checkCUDA(cudaMalloc(&backward_X_col, k_dim * cols_bwd * sizeof(float)));
		checkCUDA(cudaMalloc(&backward_X_col_T, cols_bwd * k_dim * sizeof(float)));
		checkCUDA(cudaMalloc(&backward_W_T, k_dim * m_dim * sizeof(float)));
		checkCUDA(cudaMalloc(&backward_dX_col, k_dim * cols_bwd * sizeof(float)));
	}

	// 1. Permute Gradients
	int total_output_elements = derivatives.size();
	permute_nchw_to_cnhw<<<(total_output_elements + 255)/256, 256>>>(
		derivatives.data(), backward_dY_permute,
		N, C_out, H_out * W_out
	);

	// 2. Bias Gradients
	reduce_sum_rows<<<(M_sz + 255)/256, 256>>>(
		backward_dY_permute, grad_biases.data(), M_sz, L_sz
	);

	// 3. Weight Gradients
	// 3a. Re-compute Im2Col
	dim3 grid_im2col((L_sz + 255) / 256);
	conv2d_im2col<<<grid_im2col, 256>>>(
		cached_input.data(), backward_X_col,
		N, C_in, H_in, W_in,
		filter_size, padding, stride,
		H_out, W_out
	);
	
	// 3b. Transpose X_col
	dim3 block_dim(Kernel::BLOCK_SIZE, Kernel::BLOCK_SIZE);
	dim3 grid_trans_x((L_sz + block_dim.x - 1) / block_dim.x, (K_sz + block_dim.y - 1) / block_dim.y);
	Kernel::matrix_transpose<<<grid_trans_x, block_dim>>>(backward_X_col, backward_X_col_T, K_sz, L_sz);

	// 3c. GEMM for Weights
	dim3 grid_gemm_w((K_sz + block_dim.x - 1) / block_dim.x, (M_sz + block_dim.y - 1) / block_dim.y);
	Kernel::matrix_multiply<<<grid_gemm_w, block_dim>>>(
		backward_dY_permute, backward_X_col_T, grad_weights.data(),
		M_sz, K_sz, L_sz
	);

	// 4. Input Gradients
	// 4a. Transpose Weights
	dim3 grid_trans_w((K_sz + block_dim.x - 1) / block_dim.x, (M_sz + block_dim.y - 1) / block_dim.y);
	Kernel::matrix_transpose<<<grid_trans_w, block_dim>>>(weights.data(), backward_W_T, M_sz, K_sz);

	// 4b. GEMM for Input
	dim3 grid_gemm_in((L_sz + block_dim.x - 1) / block_dim.x, (K_sz + block_dim.y - 1) / block_dim.y);
	Kernel::matrix_multiply<<<grid_gemm_in, block_dim>>>(
		backward_W_T, backward_dY_permute, backward_dX_col,
		K_sz, L_sz, M_sz
	);

	// 4c. Col2Im
	Tensor<Device::GPU> grad_input(N, C_in, H_in, W_in);
	int total_in_elements = grad_input.size(); 
	dim3 grid_col2im((total_in_elements + 255) / 256);
	conv2d_col2im<<<grid_col2im, 256>>>(
		backward_dX_col, grad_input.data(),
		N, C_in, H_in, W_in,
		filter_size, stride, padding,
		H_out, W_out
	);
	
	return grad_input;
}

}