#include <fstream>
#include <stdexcept>

#include "utils/logger.hpp"

template <typename Tag, typename Optimizer, typename Loss>
void Trainer::fit(Autoencoder<Tag> &model, Dataset &dataset, Optimizer &optimizer, Loss loss) {
	LOG("Starting training for", epochs, "epochs with batch size", batch_size);
	log_file.open(log_file_path, std::ios::out);
	log_file << "epoch,train_loss,val_loss,seconds_per_epoch" << std::endl;
	
	Autoencoder<Tag> best_model = model;
	float best_val_loss = std::numeric_limits<float>::max();

	for(size_t epoch = 1; epoch <= epochs; ++epoch) {
		timer.start<Tag>("epoch");
		
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

		timer.stop<Tag>("epoch");

		float val_loss = 0.0f;
		size_t val_batches = dataset.test_size() / batch_size;

		for(size_t i = 0; i < val_batches; ++i) {
			dataset.get_test_batch(i, batch_size, input);
			Tensor<Tag> output = model.forward(input);
			val_loss += loss(output, input);
		}
		
		float epoch_time = timer.elapsed<Tag>("epoch");
		
		float avg_train_loss = total_loss / float(batches);
		float avg_val_loss = val_loss / float(val_batches);

		LOG("Epoch", epoch, "| Time:", epoch_time / 1000.0, "s | Train Loss:", avg_train_loss, "| Val Loss:", avg_val_loss);
		log_file << epoch << "," << avg_train_loss << "," << avg_val_loss << "," << epoch_time / 1000.0 << std::endl;

		if(avg_val_loss < best_val_loss) {
			best_val_loss = avg_val_loss;
			best_model = model;
		}
	}

	model = best_model;
	log_file.close();
	LOG("Training completed");
}