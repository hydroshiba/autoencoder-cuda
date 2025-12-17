#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "config.hpp"
#include "utils/device.hpp"

template <typename Tag>
class Tensor {
	static_assert(Device::IsTag_v<Tag>, "Tensor type must be Device::CPU or Device::GPU");

private:
	size_t batches_, channels_, height_, width_;
	float* data_;

	void clean_up();

public:
	// Constructors
	Tensor();
	Tensor(size_t batches, size_t channels, size_t height, size_t width);
	
	Tensor(const Tensor& other);
	Tensor(Tensor&& other) noexcept;

	Tensor& operator=(const Tensor& other);
	Tensor& operator=(Tensor&& other) noexcept;

	template <typename T> Tensor(const Tensor<T>& other);
	template <typename T> Tensor& operator=(const Tensor<T>& other);

	// Getters
	size_t batches() const;
	size_t channels() const;
	size_t height() const;
	size_t width() const;
	size_t size() const;

	float* data();
	const float* data() const;

	float& operator()(size_t n, size_t c, size_t h, size_t w);
	const float& operator()(size_t n, size_t c, size_t h, size_t w) const;
	
	// Utilities
	void fill(float value);
	void distribute(float mean, float std_dev);

	template <typename Function>
	void transform(Function func);

	// Operators
	Tensor& operator+=(const Tensor &other);
	Tensor& operator-=(const Tensor &other);
	Tensor& operator*=(float scalar);

	Tensor operator+(const Tensor &other) const;
	Tensor operator-(const Tensor &other) const;
	Tensor operator*(float scalar) const;

	~Tensor();
	template <typename T> friend class Tensor;
};

#include "tensor.tpp"

#endif // TENSOR_HPP