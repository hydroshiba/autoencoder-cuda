#include <fstream>
#include <stdexcept>

#include "utils/logger.hpp"

template <typename Tag, typename Optimizer, typename Loss>
void Trainer::fit(Autoencoder<Tag> &model, Dataset &dataset, Optimizer &optimizer, Loss loss) {
	LOG("Starting training for", epochs, "epochs with batch size", batch_size);
	
	Autoencoder<Tag> best_model = model;
	float best_val_loss = std::numeric_limits<float>::max();

	for(size_t epoch = 1; epoch <= epochs; ++epoch) {
		timer.start("epoch");
		
		Tensor<Tag> input;
		float total_loss = 0.0f;
		size_t batches = dataset.train_size() / batch_size;

		for(size_t i = 0; i < batches; ++i) {
			model.clear_gradients();
			dataset.get_batch(i, batch_size, input);

			Tensor<Tag> output = model.forward(input);
			Tensor<Tag> grad = loss.backward(output, input);

			float loss_val = loss(output, input);
			total_loss += loss_val;

			model.backward(grad);
			optimizer.step(model, learning_rate);

			// Logging
			if(checkpoint_interval > 0 && (i + 1) % checkpoint_interval == 0)
				LOG("Epoch", epoch, "Batch", i + 1, "Loss:", loss_val);
		}

		timer.stop("epoch");

		float val_loss = 0.0f;
		size_t val_batches = dataset.test_size() / batch_size;

		for(size_t i = 0; i < val_batches; ++i) {
			dataset.get_test_batch(i, batch_size, input);
			Tensor<Tag> output = model.forward(input);
			val_loss += loss(output, input);
		}
		
		timer.synchronize();
		float epoch_time = timer.elapsed("epoch");
		
		float avg_train_loss = total_loss / float(batches);
		float avg_val_loss = val_loss / float(val_batches);

		LOG("Epoch", epoch, "| Time:", epoch_time, "ms | Train Loss:", avg_train_loss, "| Val Loss:", avg_val_loss);

		if(avg_val_loss < best_val_loss) {
			best_val_loss = avg_val_loss;
			best_model = model;
		}
	}

	model = best_model;
	LOG("Training completed");
}