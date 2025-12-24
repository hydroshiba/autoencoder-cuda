#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// A header for general-purpose CUDA kernels

namespace Kernel {

static __global__ void reduce_sum(const float* data, float* result, size_t size) {
	__shared__ float shared_data[256];
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

}

#endif // KERNEL_CUH