#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "utils/device.hpp"

#define TENSOR_BLOCK_W 16
#define TENSOR_BLOCK_H 16

template <typename Tag>
class Tensor {
	static_assert(Device::IsTag_v<Tag>, "Tensor type must be Device::CPU or Device::GPU");

private:
	size_t batches_, channels_, height_, width_;
	float* data_;

	void clean_up();

public:
	Tensor();
	Tensor(size_t batches, size_t channels, size_t height, size_t width);

	Tensor(const Tensor& other);
	Tensor(Tensor&& other) noexcept;
	Tensor& operator=(const Tensor& other);
	Tensor& operator=(Tensor&& other) noexcept;

	template <typename T> Tensor(const Tensor<T>& other);
	template <typename T> Tensor& operator=(const Tensor<T>& other);

	size_t batches() const;
	size_t channels() const;
	size_t height() const;
	size_t width() const;
	size_t size() const;

	float* data();
	const float* data() const;

	float& operator()(size_t n, size_t c, size_t h, size_t w);
	const float& operator()(size_t n, size_t c, size_t h, size_t w) const;

	template <int BLOCK_W = TENSOR_BLOCK_W, int BLOCK_H = TENSOR_BLOCK_H>
	void fill(float value);

	template <int BLOCK_W = TENSOR_BLOCK_W, int BLOCK_H = TENSOR_BLOCK_H>
	void distribute(float mean, float std_dev, uint64_t seed = 0);

	template <int BLOCK_W = TENSOR_BLOCK_W, int BLOCK_H = TENSOR_BLOCK_H, typename Function>
	void transform(Function func);

	~Tensor();
	template <typename T> friend class Tensor;
};

#include "tensor.tpp"

#endif // TENSOR_HPP