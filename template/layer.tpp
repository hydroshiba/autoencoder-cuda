#include <cmath>

namespace Layer {

// Weighted layer generic method implementations

template <typename Tag>
std::vector<Tensor<Tag>*> Weighted<Tag>::parameters() {
	return {&weights, &biases};
}

template <typename Tag>
std::vector<Tensor<Tag>*> Weighted<Tag>::gradients() {
	return {&grad_weights, &grad_biases};
}

template <typename Tag>
void Weighted<Tag>::clear_gradients() {
	grad_weights.fill(0.0f);
	grad_biases.fill(0.0f);
}

// Convolutional 2D layer generic method implementations

template <typename Tag>
Conv2D<Tag>::Conv2D(int in_channels, int out_channels, int filter_size, int stride, int padding, Activation::Func activation) :
	in_channels(in_channels),
	out_channels(out_channels),
	filter_size(filter_size),
	stride(stride),
	padding(padding)
{
	this->activation = activation;

	// Kaiming Initialization
	float fan_in = in_channels * filter_size * filter_size;
	float std_dev = std::sqrt(2.0f / fan_in);
	
	this->weights = Tensor<Tag>(out_channels, in_channels, filter_size, filter_size);
	this->weights.distribute(0.0f, std_dev); 

	this->biases = Tensor<Tag>(1, out_channels, 1, 1);
	this->biases.fill(0.0f);

	this->grad_weights = Tensor<Tag>(out_channels, in_channels, filter_size, filter_size);
	this->grad_biases = Tensor<Tag>(1, out_channels, 1, 1);
}

template <typename Tag>
std::unique_ptr<Base<Tag>> Conv2D<Tag>::clone() const {
	return std::make_unique<Conv2D<Tag>>(*this);
}

// Max Pooling 2D layer generic method implementations

template <typename Tag>
MaxPool2D<Tag>::MaxPool2D(int size, int str, Activation::Func act) :
	pool_size(size),
	stride(str) 
{
	this->activation = act;
}

template <typename Tag>
std::unique_ptr<Base<Tag>> MaxPool2D<Tag>::clone() const {
	return std::make_unique<MaxPool2D<Tag>>(*this);
}

// Upsample 2D generic method implementations

template <typename Tag>
UpSample2D<Tag>::UpSample2D(int s, Activation::Func act) : scale(s) {
	this->activation = act;
}

template <typename Tag>
std::unique_ptr<Base<Tag>> UpSample2D<Tag>::clone() const {
	return std::make_unique<UpSample2D<Tag>>(*this);
}

}