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

// Specific template implementations for templated methods

template <typename Function>
__global__ void transform_kernel(float* data, size_t size, Function func) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) data[i] = func(data[i]);
}

template <typename Tag>
template <typename Function>
void Tensor<Tag>::transform(Function func) {
	if constexpr (std::is_same_v<Tag, Device::CPU>) {
		for(size_t i = 0; i < size(); ++i) {
			data_[i] = func(data_[i]);
		}
	}
	else {
		size_t total_size = size();
		size_t threads = Config::Tensor::block_width * Config::Tensor::block_height;
		size_t blocks = (total_size + threads - 1) / threads;

		transform_kernel<<<blocks, threads>>>(data_, total_size, func);
		// cudaDeviceSynchronize();
		checkCUDA(cudaGetLastError());
	}
}

template <typename Tag>
Tensor<Tag> Tensor<Tag>::operator+(const Tensor &other) const {
	Tensor<Tag> result = *this;
	result += other;
	return result;
}

template <typename Tag>
Tensor<Tag> Tensor<Tag>::operator-(const Tensor &other) const {
	Tensor<Tag> result = *this;
	result -= other;
	return result;
}

template <typename Tag>
Tensor<Tag> Tensor<Tag>::operator*(float scalar) const {
	Tensor<Tag> result = *this;
	result *= scalar;
	return result;
}

template <typename Tag>
Tensor<Tag> operator*(float scalar, const Tensor<Tag> &tensor) {
	return tensor * scalar;
}