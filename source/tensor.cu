#include "tensor.hpp"
#include <kernel.cuh>

Tensor &Tensor::operator=(const Tensor &other) {
	if (this == &other) return *this;

	// We don't care about device data, only copy host data for consistency

	host_data = other.host_data;
	N = other.N;
	C = other.C;
	H = other.H;
	W = other.W;

	if(device_data) {
		CHECK(cudaFree(device_data));
		device_data = nullptr;
		on_gpu = false;
	}

	return *this;
}

void Tensor::clear_tensor() {
	if (on_gpu) {
		cudaMemset(device_data, 0, host_data.size() * sizeof(float));
	}
	else {
		std::fill(host_data.begin(), host_data.end(), 0.0f);	
	}
}

Tensor::~Tensor() {
	if(device_data) {
		CHECK(cudaFree(device_data));
		device_data = nullptr;
	}
}

Tensor& Tensor::to_gpu() {
	if(on_gpu) return *this;
	on_gpu = true;

	CHECK(cudaMalloc(&device_data, host_data.size() * sizeof(float)));
	CHECK(cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice));
	return *this;
}

Tensor& Tensor::to_cpu() {
	if(!on_gpu) return *this;
	on_gpu = false;

	cudaMemcpy(host_data.data(), device_data, host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
	
	CHECK(cudaFree(device_data));
	device_data = nullptr;
	
	return *this;
}