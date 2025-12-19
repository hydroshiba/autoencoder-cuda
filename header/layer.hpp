#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <memory>

#include "config.hpp"
#include "tensor.hpp"
#include "activation.cuh"

namespace Layer {

template <typename Tag>
class Base {
	static_assert(Device::IsTag_v<Tag>, "Layer type must be Device::CPU or Device::GPU");

protected:
	Tensor<Tag> cached_input;
	Activation::Func activation = Activation::Identity();

public:
	virtual Tensor<Tag> forward(const Tensor<Tag> &input) = 0;
	virtual Tensor<Tag> backward(const Tensor<Tag> &grad_output) = 0;

	virtual std::vector<Tensor<Tag>*> parameters() { return {}; }
	virtual std::vector<Tensor<Tag>*> gradients() { return {}; }

	virtual void clear_gradients() {}
	virtual std::unique_ptr<Base> clone() const = 0;
	virtual ~Base() = default;

	// For fuck sake, kill yourself NVCC
	template <typename Activation>
	void forward_activate(Tensor<Tag> &tensor, Activation act) {
		tensor.transform([act] __device__ (float x) { return act.forward(x); });
	}

	template <typename Activation>
	void backward_activate(Tensor<Tag> &tensor, Activation act) {
		tensor.transform([act] __device__ (float x) { return act.backward(x); });
	}
};

template <typename Tag>
class Weighted : public Base<Tag> {
protected:
	Tensor<Tag> weights;
	Tensor<Tag> biases;

	Tensor<Tag> grad_weights;
	Tensor<Tag> grad_biases;

public:
	std::vector<Tensor<Tag>*> parameters() override;
	std::vector<Tensor<Tag>*> gradients() override;

	void clear_gradients() override;
	virtual ~Weighted() = default;
};

template <typename Tag>
class Conv2D : public Weighted<Tag> {
private:
	int in_channels, out_channels;
	int filter_size, stride, padding;

public:
	Conv2D(
		int in_channels,
		int out_channels,
		int filter_size,
		int stride = 1,
		int padding = 0,
		Activation::Func activation = Activation::Identity()
	);

	Tensor<Tag> forward(const Tensor<Tag> &input) override;
	Tensor<Tag> backward(const Tensor<Tag> &grad_output) override;
	std::unique_ptr<Base<Tag>> clone() const override;
};

template <typename Tag>
class MaxPool2D : public Base<Tag> {
private:
	int pool_size;
	int stride;
	Tensor<Tag> mask;

public:
	MaxPool2D(int pool_size = 2, int stride = 2, Activation::Func activation = Activation::Identity());

	Tensor<Tag> forward(const Tensor<Tag> &input) override;
	Tensor<Tag> backward(const Tensor<Tag> &grad_output) override;
	std::unique_ptr<Base<Tag>> clone() const override;
};

template <typename Tag>
class UpSample2D : public Base<Tag> {
private:
	int scale;

public:
	UpSample2D(int scale = 2, Activation::Func activation = Activation::Identity());

	Tensor<Tag> forward(const Tensor<Tag> &input) override;
	Tensor<Tag> backward(const Tensor<Tag> &grad_output) override;
	std::unique_ptr<Base<Tag>> clone() const override;
};

}

#include "layer.tpp"

#endif // LAYER_HPP