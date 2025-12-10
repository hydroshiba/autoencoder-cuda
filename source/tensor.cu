#include "tensor.hpp"

Tensor& Tensor::to_gpu() {
	if(on_gpu) return *this;
	on_gpu = true;

	cudaMalloc(&device_data, host_data.size() * sizeof(float));
	cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

Tensor& Tensor::to_cpu() {
	if(!on_gpu) return *this;
	on_gpu = false;

	cudaMemcpy(host_data.data(), device_data, host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(device_data);
}