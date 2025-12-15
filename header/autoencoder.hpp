#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include <vector>
#include <memory>

#include "layer.hpp"

namespace Autoencoder {

class Base {
	protected:
		std::vector<std::unique_ptr<Layer>> layers;
		int encode_layer = 0;

		void add_layer(std::unique_ptr<Layer> layer);
		
		virtual Tensor forward_encode(const Tensor &input) = 0;
		virtual Tensor forward_decode(const Tensor &input) = 0;

		virtual Tensor backward_decode(const Tensor &gradient) = 0;
		virtual Tensor backward_encode(const Tensor &gradient) = 0;

	public:
		Base();
		void build();

		Tensor forward(const Tensor &input);
		Tensor encode(const Tensor &input);
		void backward(const Tensor &gradient);

		virtual void update(float learning_rate) = 0;

		void save_model(const std::string &file_path);
		void load_model(const std::string &file_path);
};

class CPU: public Base {
	private:
		Tensor forward_encode(const Tensor &input) override;
		Tensor forward_decode(const Tensor &input) override;

		Tensor backward_decode(const Tensor &gradient) override;
		Tensor backward_encode(const Tensor &gradient) override;
	public:
		void update(float learning_rate) override;
	};

class GPU: public Base {
	private:
		Tensor forward_encode(const Tensor &input) override;
		Tensor forward_decode(const Tensor &input) override;

		Tensor backward_decode(const Tensor &gradient) override;
		Tensor backward_encode(const Tensor &gradient) override;
	public:
		GPU();
		//GPU(const Base &base);

		void update(float learning_rate) override;
};

}

#endif