#ifndef LOSS_CUH
#define LOSS_CUH

#include <numeric>
#include <cuda_runtime.h>

#include "config.hpp"
#include "tensor.hpp"
#include "utils/kernel.cuh"

namespace Loss {

class MSE {
public:
	template <typename Tag>
	float operator()(const Tensor<Tag> &predicted, const Tensor<Tag> &target) const {
		if(predicted.batches() != target.batches() ||
		   predicted.channels() != target.channels() ||
		   predicted.height() != target.height() ||
		   predicted.width() != target.width()) {
			throw std::invalid_argument("MSE Loss: Predicted and target tensors must have the same shape.");
		}

		size_t elements = predicted.size();
		Tensor<Tag> diff = predicted - target; // Element-wise difference
		diff.transform([] __host__ __device__ (float x) { return x * x; }); // Difference squared
		
		float loss = 0.0f;

		if constexpr (std::is_same_v<Tag, Device::CPU>) loss = std::reduce(diff.data(), diff.data() + elements, 0.0f);
		else {
			float* gpu_loss;
			checkCUDA(cudaMalloc(&gpu_loss, sizeof(float)));
			checkCUDA(cudaMemset(gpu_loss, 0, sizeof(float)));

			size_t threads = Config::Tensor::block_width * Config::Tensor::block_height;
			size_t blocks = (elements + threads - 1) / threads;

			Kernel::reduce_sum<<<blocks, threads, threads * sizeof(float)>>>(diff.data(), gpu_loss, elements);
			checkCUDA(cudaGetLastError());
			checkCUDA(cudaDeviceSynchronize());

			checkCUDA(cudaMemcpy(&loss, gpu_loss, sizeof(float), cudaMemcpyDeviceToHost));
			checkCUDA(cudaFree(gpu_loss));
		}

		loss /= static_cast<float>(elements);
		return loss;
	}
};

}

#endif // LOSS_CUH