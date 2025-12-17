#include "tensor.hpp"
#include "utils/error.cuh"
#include "utils/random.cuh"

// Tensor GPU kernels

__global__ void fill_kernel(float* data, size_t size, float value) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) data[i] = value;
}

__global__ void distribute_kernel(float* data, size_t size, float mean, float std_dev, uint64_t seed) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) {
		uint64_t state = seed + i;
		uint64_t rands[2] = {Random::splitmix64(state), Random::splitmix64(state)};
		data[i] = (Random::box_muller(rands) * std_dev) + mean;
	}
}

// Tensor GPU specialization implementations

template <>
Tensor<Device::GPU>::Tensor(size_t batches, size_t channels, size_t height, size_t width) :
	batches_(batches),
	channels_(channels),
	height_(height),
	width_(width),
	data_(nullptr)
{
	size_t total_size = size();
	checkCUDA(cudaMalloc(&data_, total_size * sizeof(float)));
}

template <>
template<>
Tensor<Device::GPU>::Tensor(const Tensor<Device::CPU>& other) :
	batches_(other.batches_),
	channels_(other.channels_),
	height_(other.height_),
	width_(other.width_),
	data_(nullptr)
{
	size_t total_size = other.size();
	checkCUDA(cudaMalloc(&data_, total_size * sizeof(float)));
	checkCUDA(cudaMemcpy(data_, other.data_, total_size * sizeof(float), cudaMemcpyHostToDevice));
}

template <>
Tensor<Device::GPU>::Tensor(const Tensor<Device::GPU>& other) :
	batches_(other.batches_),
	channels_(other.channels_),
	height_(other.height_),
	width_(other.width_),
	data_(nullptr)
{
	size_t total_size = other.size();
	checkCUDA(cudaMalloc(&data_, total_size * sizeof(float)));
	checkCUDA(cudaMemcpy(data_, other.data_, total_size * sizeof(float), cudaMemcpyDeviceToDevice));
}

template <>
void Tensor<Device::GPU>::clean_up() {
	if(data_) checkCUDA(cudaFree(data_), false);
	data_ = nullptr;
}

template <>
template <>
Tensor<Device::GPU>& Tensor<Device::GPU>::operator=(const Tensor<Device::CPU>& other) {
	if(this->size() != other.size()) {
		this->clean_up();
		checkCUDA(cudaMalloc(&data_, other.size() * sizeof(float)));
	}

	batches_ = other.batches_;
	channels_ = other.channels_;
	height_ = other.height_;
	width_ = other.width_;
	checkCUDA(cudaMemcpy(data_, other.data_, other.size() * sizeof(float), cudaMemcpyHostToDevice));
	
	return *this;
}

template <>
Tensor<Device::GPU>& Tensor<Device::GPU>::operator=(const Tensor<Device::GPU>& other) {
	if(this == &other) return *this;

	if(this->size() != other.size()) {
		this->clean_up();
		checkCUDA(cudaMalloc(&data_, other.size() * sizeof(float)));
	}

	batches_ = other.batches_;
	channels_ = other.channels_;
	height_ = other.height_;
	width_ = other.width_;
	checkCUDA(cudaMemcpy(data_, other.data_, other.size() * sizeof(float), cudaMemcpyDeviceToDevice));

	return *this;
}

template <>
void Tensor<Device::GPU>::fill(float value) {
	size_t total_size = size();
	size_t threads = Config::Tensor::block_width * Config::Tensor::block_height;
	size_t blocks = (total_size + threads - 1) / threads;

	fill_kernel<<<blocks, threads>>>(data_, total_size, value);
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());
}

template <>
void Tensor<Device::GPU>::distribute(float mean, float std_dev, uint64_t seed) {
	size_t total_size = size();
	size_t threads = Config::Tensor::block_width * Config::Tensor::block_height;
	size_t blocks = (total_size + threads - 1) / threads;

	distribute_kernel<<<blocks, threads>>>(data_, total_size, mean, std_dev, seed);
	checkCUDA(cudaGetLastError());
	checkCUDA(cudaDeviceSynchronize());
}

// Disable GPU memory acess from Host

template <>
inline float& Tensor<Device::GPU>::operator()(size_t n, size_t c, size_t h, size_t w) {
	throw std::runtime_error("Cannot access GPU memory from Host");
}

template <>
inline const float& Tensor<Device::GPU>::operator()(size_t n, size_t c, size_t h, size_t w) const {
	throw std::runtime_error("Cannot access GPU memory from Host");
}