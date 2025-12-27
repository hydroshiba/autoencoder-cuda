#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// A header for general-purpose CUDA kernels

namespace Kernel {

const int BLOCK_SIZE = 16;

static __global__ void reduce_sum(const float* data, float* result, size_t size) {
	__shared__ float shared_data[BLOCK_SIZE * BLOCK_SIZE];
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	
	shared_data[tid] = (i < size) ? data[i] : 0.0f;
	__syncthreads();

	for(size_t s = blockDim.x / 2; s > 0; s >>= 1) {
		if(tid < s) shared_data[tid] += shared_data[tid + s];
		__syncthreads();
	}

	if(tid == 0) atomicAdd(result, shared_data[0]);
}

static __global__ void vector_multiply(float* a, const float* b, size_t size) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size) a[idx] *= b[idx];
}

// Matrix-related kernels

static __global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

	if(x < cols && y < rows)
		tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

	__syncthreads();

	x = blockIdx.y * blockDim.x + threadIdx.x;
	y = blockIdx.x * blockDim.y + threadIdx.y;

	if(x < rows && y < cols)
		output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

static __global__ void matrix_multiply(
	const float *A, const float *B, float *C,
	int M, int N, int K)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * BLOCK_SIZE + ty;
	int col = blockIdx.x * BLOCK_SIZE + tx;

	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	float sum = 0.0f;

	for(int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

		int tiled_k_idx = t * BLOCK_SIZE;

		if(row < M && (tiled_k_idx + tx) < K) As[ty][tx] = A[row * K + (tiled_k_idx + tx)];
		else As[ty][tx] = 0.0f;

		if(col < N && (tiled_k_idx + ty) < K) Bs[ty][tx] = B[(tiled_k_idx + ty) * N + col];
		else Bs[ty][tx] = 0.0f;

		__syncthreads();

		#pragma unroll
		for(int k = 0; k < BLOCK_SIZE; ++k)
			sum += As[ty][k] * Bs[k][tx];
		
		// Synchronize to ensure computation is done before loading next tile
		__syncthreads(); 
	}

	if(row < M && col < N) C[row * N + col] = sum;
}

}

#endif // KERNEL_CUH