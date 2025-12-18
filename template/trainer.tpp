#include <fstream>
#include <stdexcept>

#include "utils/logger.hpp"

template <typename Tag, typename Optimizer, typename Loss>
void Trainer::fit(Autoencoder<Tag> &model, Dataset &dataset, Optimizer &optimizer, Loss loss) {
	LOG("Starting training for", epochs, "epochs with batch size", batch_size);

	for(size_t epoch = 0; epoch < epochs; ++epoch) {
		timer.start("epoch");
		
		float total_loss = 0.0f;
		size_t batches = dataset.train_size() / Config::batch_size;

		for(size_t i = 0; i < batches; ++i) {
			Tensor<Tag> input = dataset.get_batch(i, Config::batch_size);
			Tensor<Tag> output = model.forward(input);

			float loss_val = loss(output, input);
			total_loss += loss_val;

			// d/dx (x - t)^2 = 2(x - t)
			Tensor<Tag> grad = output - input;
			float scale = 2.0f / static_cast<float>(output.size());
			grad.transform([=] __host__ __device__ (float x) { return x * scale; });

			model.backward(grad);
			optimizer.step(model, Config::learning_rate);

			// Logging
			LOG("Epoch", epoch + 1, "Batch", i, "Loss:", loss_val);
		}

		timer.stop("epoch");
		timer.synchronize();
		
		float epoch_time = timer.elapsed("epoch");
		float avg_loss = total_loss / batches;
		LOG("Epoch", epoch + 1, "| Loss:", avg_loss, "| Time:", epoch_time, "ms");
	}

	LOG("Training completed");
}