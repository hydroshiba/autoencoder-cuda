#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

#include "layer.hpp"

namespace Autoencoder
{

	class Base
	{
	protected:
		std::vector<std::unique_ptr<Layer>> layers;
		int encode_layer = 0;

		void add_layer(std::unique_ptr<Layer> layer);

		virtual Tensor forward_encode(const Tensor &input, cudaStream_t stream = 0) = 0;
		virtual Tensor forward_decode(const Tensor &input, cudaStream_t stream = 0) = 0;

		virtual Tensor backward_decode(const Tensor &gradient, cudaStream_t stream = 0) = 0;
		virtual Tensor backward_encode(const Tensor &gradient, cudaStream_t stream = 0) = 0;

	public:
		Base();
		void build();

		Tensor forward(const Tensor &input);
		Tensor encode(const Tensor &input);
		Tensor decode(const Tensor &input);
		void backward(const Tensor &gradient);

		virtual void update(float learning_rate) = 0;

		void save_model(const std::string &file_path);
		void load_model(const std::string &file_path);
	};

	class CPU : public Base
	{
	private:
		Tensor forward_encode(const Tensor &input, cudaStream_t stream = 0) override;
		Tensor forward_decode(const Tensor &input, cudaStream_t stream = 0) override;

		Tensor backward_decode(const Tensor &gradient, cudaStream_t stream = 0) override;
		Tensor backward_encode(const Tensor &gradient, cudaStream_t stream = 0) override;

	public:
		void update(float learning_rate) override;
	};

	class GPU : public Base
	{
	private:
		Tensor forward_encode(const Tensor &input, cudaStream_t stream = 0) override;
		Tensor forward_decode(const Tensor &input, cudaStream_t stream = 0) override;

		Tensor backward_decode(const Tensor &gradient, cudaStream_t stream = 0) override;
		Tensor backward_encode(const Tensor &gradient, cudaStream_t stream = 0) override;

	public:
		GPU();
		// GPU(const Base &base);

		void update(float learning_rate) override;
	};

}

#endif