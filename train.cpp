#include <filesystem>

#include "dataset.hpp"
#include "optimizer.hpp"
#include "loss.cuh"

#include "autoencoder.hpp"
#include "trainer.hpp"

#include "utils/logger.hpp"

template <typename Tag>
void train(const std::string& checkpoint_path) {
	Autoencoder<Tag> model;
	Trainer trainer;
	Dataset dataset("content/cifar-10-batches-bin");
	Optimizer::SGD optimizer;

	trainer.fit(model, dataset, optimizer, Loss::MSE());
	model.save(checkpoint_path);
}

int main(int argc, char** argv) {
	LOG("Starting Autoencoder Training Pipeline");
	std::string device = "gpu";

	if(argc > 1) {
		device = argv[1];
		if(device == "cpu") LOG("Using CPU for training");
		else if(device == "gpu") LOG("Using GPU for training");
		else LOG("Unknown argument:", device, "| Defaulting to GPU for training");
	}

	std::string config_path = argc > 2 ? argv[2] : "config.yaml";
	Config::load(config_path);

	std::string checkpoint_path = "content/model/" + device + "_autoencoder.dat";
	std::filesystem::create_directories("content/model/");
	
	if(device == "cpu") train<Device::CPU>(checkpoint_path);
	else train<Device::GPU>(checkpoint_path);

	LOG("Training completed");
	LOG("Model saved to", checkpoint_path);
}