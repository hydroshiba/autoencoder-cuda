#include "tensor.hpp"
#include "utils/error.cuh"
#include "utils/random.cuh"

// Tensor CPU specialization implementations

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
	if(this == &other) return *this;

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

template <>
void Tensor<Device::CPU>::fill(float value) {
	std::fill(data_, data_ + size(), value);
}

template <>
void Tensor<Device::CPU>::distribute(float mean, float std_dev) {
	static size_t call_offset = 0;
	uint64_t seed = static_cast<uint64_t>(Config::seed) + (call_offset++ * 0x9E3779B97F4A7C15ULL);
	
	// Initialize state ONCE outside the loop to create a continuous stream
	uint64_t state = seed;
	uint64_t s0 = Random::splitmix64(state);
	uint64_t s1 = Random::splitmix64(state);
	uint64_t rands[2] = {s0, s1};
	
	for(size_t i = 0; i < size(); ++i) {
		data_[i] = (Random::box_muller(rands) * std_dev) + mean;
	}
}

template <>
Tensor<Device::CPU>& Tensor<Device::CPU>::operator+=(const Tensor &other) {
	if(this->batches_ != other.batches_ ||
	   this->channels_ != other.channels_ ||
	   this->height_ != other.height_ ||
	   this->width_ != other.width_)
		throw std::invalid_argument("Tensor shapes do not match for addition");

	for(size_t i = 0; i < this->size(); ++i) this->data_[i] += other.data_[i];
	return *this;
}

template <>
Tensor<Device::CPU>& Tensor<Device::CPU>::operator-=(const Tensor &other) {
	if(this->batches_ != other.batches_ ||
	   this->channels_ != other.channels_ ||
	   this->height_ != other.height_ ||
	   this->width_ != other.width_)
		throw std::invalid_argument("Tensor shapes do not match for subtraction");

	for(size_t i = 0; i < this->size(); ++i) this->data_[i] -= other.data_[i];
	return *this;
}

template <>
Tensor<Device::CPU>& Tensor<Device::CPU>::operator*=(float scalar) {
	for(size_t i = 0; i < this->size(); ++i) this->data_[i] *= scalar;
	return *this;
}