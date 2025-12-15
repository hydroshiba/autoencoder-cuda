#include "tensor.hpp"
#include "utils/error.cuh"
#include "utils/random.cuh"

// CPU specialization implementations

template <>
Tensor<Device::CPU>::Tensor(size_t batches, size_t channels, size_t height, size_t width) :
	batches_(batches),
	channels_(channels),
	height_(height),
	width_(width),
	data_(new float[batches * channels * height * width]) {}

template <>
Tensor<Device::CPU>::Tensor(const Tensor<Device::CPU>& other) :
	batches_(other.batches_),
	channels_(other.channels_),
	height_(other.height_),
	width_(other.width_),
	data_(new float[other.size()])
{
	std::copy(other.data_, other.data_ + other.size(), data_);
}

template<>
template<>
Tensor<Device::CPU>::Tensor(const Tensor<Device::GPU>& other) :
	batches_(other.batches_),
	channels_(other.channels_),
	height_(other.height_),
	width_(other.width_)
{
	size_t total_size = other.size();
	data_ = new float[total_size];
	checkCUDA(cudaMemcpy(data_, other.data_, total_size * sizeof(float), cudaMemcpyDeviceToHost));
}

template <>
void Tensor<Device::CPU>::clean_up() {
	if(data_) delete[] data_;
	data_ = nullptr;
}

template <>
Tensor<Device::CPU>& Tensor<Device::CPU>::operator=(const Tensor<Device::CPU>& other) {
	if (this == &other) return *this;

	if(this->size() != other.size()) {
		this->clean_up();
		data_ = new float[other.size()];
	}

	batches_ = other.batches_;
	channels_ = other.channels_;
	height_ = other.height_;
	width_ = other.width_;
	std::copy(other.data_, other.data_ + other.size(), data_);

	return *this;
}

template <>
template<>
Tensor<Device::CPU>& Tensor<Device::CPU>::operator=(const Tensor<Device::GPU>& other) {
	if(this->size() != other.size()) {
		this->clean_up();
		data_ = new float[other.size()];
	}

	batches_ = other.batches_;
	channels_ = other.channels_;
	height_ = other.height_;
	width_ = other.width_;
	checkCUDA(cudaMemcpy(data_, other.data_, other.size() * sizeof(float), cudaMemcpyDeviceToHost));

	return *this;
}