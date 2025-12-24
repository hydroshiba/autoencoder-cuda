#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <variant>

namespace Activation {

class Identity {
public:
	__host__ __device__ float forward(float x) const {
		return x;
	}

	__host__ __device__ float backward(float x) const {
		return 1.0f;
	}
};

class ReLU {
public:
	__host__ __device__ float forward(float x) const {
		return x > 0 ? x : 0;
	}

	__host__ __device__ float backward(float x) const {
		return x > 0 ? 1.0f : 0.0f;
	}
};

class Sigmoid {
public:
	__host__ __device__ float forward(float x) const {
		return 1.0f / (1.0f + expf(-x));
	}

	__host__ __device__ float backward(float x) const {
		float y = forward(x);
		return y * (1.0f - y);
	}
};

using Func = std::variant<
	Identity,
	ReLU,
	Sigmoid
>;

}

#endif // ACTIVATION_CUH