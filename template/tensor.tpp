#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>

#include "utils/error.cuh"
#include "utils/random.cuh"

// General template implementations

template <typename Tag>
Tensor<Tag>::Tensor() :
	batches_(0), channels_(0), height_(0), width_(0),
	data_(nullptr) {}

template <typename Tag>
Tensor<Tag>::Tensor(Tensor<Tag>&& other) noexcept :
	batches_(other.batches_),
	channels_(other.channels_),
	height_(other.height_),
	width_(other.width_),
	data_(other.data_)
{
	other.batches_ = 0;
	other.channels_ = 0;
	other.height_ = 0;
	other.width_ = 0;
	other.data_ = nullptr;
}

template <typename Tag>
Tensor<Tag>& Tensor<Tag>::operator=(Tensor<Tag>&& other) noexcept {
	if (this == &other) return *this;

	this->clean_up();
	batches_ = other.batches_; other.batches_ = 0;
	channels_ = other.channels_; other.channels_ = 0;
	height_ = other.height_; other.height_ = 0;
	width_ = other.width_; other.width_ = 0;

	data_ = other.data_;
	other.data_ = nullptr;

	return *this;
}

template <typename Tag>
size_t Tensor<Tag>::batches() const { return batches_; }

template <typename Tag>
size_t Tensor<Tag>::channels() const { return channels_; }

template <typename Tag>
size_t Tensor<Tag>::height() const { return height_; }

template <typename Tag>
size_t Tensor<Tag>::width() const { return width_; }

template <typename Tag>
size_t Tensor<Tag>::size() const {
	return batches_ * channels_ * height_ * width_;
}

template <typename Tag>
float* Tensor<Tag>::data() { return data_; }

template <typename Tag>
const float* Tensor<Tag>::data() const { return data_; }

template <typename Tag>
float& Tensor<Tag>::operator()(size_t n, size_t c, size_t h, size_t w) {
	if(n >= batches_ || c >= channels_ || h >= height_ || w >= width_)
		throw std::out_of_range("Tensor index out of range");

	return data_[
		n * (channels_ * height_ * width_) +
		c * (height_ * width_) +
		h * width_ + w
	];
}

template <typename Tag>
const float& Tensor<Tag>::operator()(size_t n, size_t c, size_t h, size_t w) const {
	if(n >= batches_ || c >= channels_ || h >= height_ || w >= width_)
		throw std::out_of_range("Tensor index out of range");

	return data_[
		n * (channels_ * height_ * width_) +
		c * (height_ * width_) +
		h * width_ + w
	];
}

template <typename Tag>
Tensor<Tag>::~Tensor() {
	clean_up();
	batches_ = channels_ = height_ = width_ = 0;
}

// Specific template implementations for methods

template <typename Function>
__global__ void transform_kernel(float* data, size_t size, Function func) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) data[i] = func(data[i]);
}

__global__ void fill_kernel(float* data, size_t size, float value);
__global__ void distribute_kernel(float* data, size_t size, float mean, float std_dev, uint64_t seed);

template <typename Tag>
template <int BLOCK_W, int BLOCK_H>
void Tensor<Tag>::fill(float value) {
	if constexpr (std::is_same_v<Tag, Device::CPU>) {
		std::fill(data_, data_ + size(), value);
	}
	else {
		size_t total_size = size();

		if(value == 0.0f) checkCUDA(cudaMemset(data_, 0, total_size * sizeof(float)));
		else {
			size_t threads = BLOCK_W * BLOCK_H;
			size_t blocks = (total_size + threads - 1) / threads;

			fill_kernel<<<blocks, threads>>>(data_, total_size, value);
			cudaDeviceSynchronize();
			checkCUDA(cudaGetLastError());
		}
	}
}

template <typename Tag>
template <int BLOCK_W, int BLOCK_H>
void Tensor<Tag>::distribute(float mean, float std_dev, uint64_t seed) {
	if constexpr (std::is_same_v<Tag, Device::CPU>) {
		for(size_t i = 0; i < size(); ++i) {
			uint64_t state = seed + i;
			uint64_t rands[2] = {Random::splitmix64(state), Random::splitmix64(state)};
			data_[i] = (Random::box_muller(rands) * std_dev) + mean;
		}
	}
	else {
		size_t total_size = size();
		size_t threads = BLOCK_W * BLOCK_H;
		size_t blocks = (total_size + threads - 1) / threads;

		distribute_kernel<<<blocks, threads>>>(data_, total_size, mean, std_dev, seed);
		cudaDeviceSynchronize();
		checkCUDA(cudaGetLastError());
	}
}

template <typename Tag>
template <int BLOCK_W, int BLOCK_H, typename Function>
void Tensor<Tag>::transform(Function func) {
	if constexpr (std::is_same_v<Tag, Device::CPU>) {
		for(size_t i = 0; i < size(); ++i) {
			data_[i] = func(data_[i]);
		}
	}
	else {
		size_t total_size = size();
		size_t threads = BLOCK_W * BLOCK_H;
		size_t blocks = (total_size + threads - 1) / threads;

		transform_kernel<<<blocks, threads>>>(data_, total_size, func);
		cudaDeviceSynchronize();
		checkCUDA(cudaGetLastError());
	}
}