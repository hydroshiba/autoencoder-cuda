#include "tensor.hpp"

#include <algorithm>
#include <stdexcept>

Tensor::Tensor()
	: host_data(), device_data(nullptr), N(0), C(0), H(0), W(0), on_gpu(false)
{
}

Tensor::Tensor(int n, int c, int h, int w)
	: Tensor()
{
	resize(n, c, h, w);
}

Tensor::Tensor(const Tensor& other)
	: host_data(other.host_data), device_data(nullptr),
	  N(other.N), C(other.C), H(other.H), W(other.W), on_gpu(other.on_gpu)
{
}

Tensor& Tensor::operator=(const Tensor& other)
{
	if (this == &other)
	{
		return *this;
	}

	host_data = other.host_data;
	N = other.N;
	C = other.C;
	H = other.H;
	W = other.W;
	on_gpu = other.on_gpu;

	if (device_data)
	{
		delete[] device_data;
		device_data = nullptr;
	}

	return *this;
}

Tensor::~Tensor()
{
	if (device_data)
	{
		delete[] device_data;
		device_data = nullptr;
	}
}

void Tensor::resize(int n, int c, int h, int w)
{
	if (n < 0 || c < 0 || h < 0 || w < 0)
	{
		throw std::invalid_argument("Tensor dimensions must be non-negative");
	}

	N = n;
	C = c;
	H = h;
	W = w;

	std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(c) *
						static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
	host_data.assign(total, 0.0f);
}

int Tensor::batch() const
{
	return N;
}

int Tensor::channels() const
{
	return C;
}

int Tensor::height() const
{
	return H;
}

int Tensor::width() const
{
	return W;
}

size_t Tensor::size() const
{
	return host_data.size();
}

void Tensor::to_gpu()
{
	on_gpu = true;
}

void Tensor::to_cpu()
{
	on_gpu = false;
}

bool Tensor::is_gpu() const
{
	return on_gpu;
}

float& Tensor::operator()(int n, int c, int h, int w)
{
	return host_data.at(index(n, c, h, w));
}

const float& Tensor::operator()(int n, int c, int h, int w) const
{
	return host_data.at(index(n, c, h, w));
}

std::size_t Tensor::index(int n, int c, int h, int w) const
{
	if (n < 0 || n >= N || c < 0 || c >= C || h < 0 || h >= H || w < 0 || w >= W)
	{
		throw std::out_of_range("Tensor indices out of range");
	}

	return static_cast<std::size_t>(((n * C + c) * H + h) * W + w);
}