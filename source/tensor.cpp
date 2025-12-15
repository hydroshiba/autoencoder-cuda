#include "tensor.hpp"

#include <algorithm>
#include <stdexcept>

Tensor::Tensor() : host_data(), device_data(nullptr),
				   N(0), C(0), H(0), W(0),
				   on_gpu(false) {}

Tensor::Tensor(const Tensor &other) :
    host_data(other.host_data),
    device_data(nullptr),
    N(other.N), C(other.C), H(other.H), W(other.W),
    on_gpu(false)
{}

Tensor::Tensor(int n, int c, int h, int w) : Tensor()
{
	resize(n, c, h, w);
}

// Tensor::Tensor(const Tensor &other) : host_data(other.host_data), device_data(nullptr),
// 									  N(other.N), C(other.C), H(other.H), W(other.W), on_gpu(other.on_gpu) {}

void Tensor::resize(int n, int c, int h, int w)
{
	if (n < 0 || c < 0 || h < 0 || w < 0)
		throw std::invalid_argument("Tensor dimensions must be non-negative");

	N = n;
	C = c;
	H = h;
	W = w;

	std::size_t total = size_t(n) * size_t(c) * size_t(h) * size_t(w);
	host_data.assign(total, 0.0f);
}

int Tensor::batch() const { return N; }

int Tensor::channels() const { return C; }

int Tensor::height() const { return H; }

int Tensor::width() const { return W; }

size_t Tensor::size() const { return host_data.size(); }

bool Tensor::is_gpu() const { return on_gpu; }

// float &Tensor::operator()(int n, int c, int h, int w)
// {
// 	if (on_gpu)
// 		return device_data[index(n, c, h, w)];
// 	return host_data.at(index(n, c, h, w));
// }

// const float &Tensor::operator()(int n, int c, int h, int w) const
// {
// 	if (on_gpu)
// 		return device_data[index(n, c, h, w)];
// 	return host_data.at(index(n, c, h, w));
// }

float &Tensor::operator()(int n, int c, int h, int w) {
    if (on_gpu)
        throw std::runtime_error("Accessing GPU tensor from CPU");
    return host_data.at(index(n, c, h, w));
}

const float &Tensor::operator()(int n, int c, int h, int w) const {
    if (on_gpu)
        throw std::runtime_error("Accessing GPU tensor from CPU");
    return host_data.at(index(n, c, h, w));
}

float *Tensor::data()
{
	if (on_gpu)
		return device_data;
	return host_data.data();
}

const float *Tensor::data() const
{
	if (on_gpu)
		return device_data;
	return host_data.data();
}

std::size_t Tensor::index(int n, int c, int h, int w) const
{
	if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W)
		throw std::out_of_range("Tensor indices out of range");

	return static_cast<std::size_t>(((n * C + c) * H + h) * W + w);
}