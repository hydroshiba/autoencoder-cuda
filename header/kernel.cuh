#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdint.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, static_cast<int>(error), cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA SGD kernel: params[i] -= lr * grads[i]
__global__ void sgd_update_kernel(float* params, const float* grads, float lr, int size);

// Host launcher for the SGD kernel
void sgd_update_device(float* params, const float* grads, float lr, std::size_t size);

// struct GpuTimer
// {
//     cudaEvent_t start;
//     cudaEvent_t stop;

//     GpuTimer()
//     {
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);
//     }

//     ~GpuTimer()
//     {
//         cudaEventDestroy(start);
//         cudaEventDestroy(stop);
//     }

//     void Start()
//     {
//         cudaEventRecord(start, 0);
//         cudaEventSynchronize(start);
//     }

//     void Stop()
//     {
//         cudaEventRecord(stop, 0);
//     }

//     float Elapsed()
//     {
//         float elapsed;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&elapsed, start, stop);
//         return elapsed;
//     }
// };