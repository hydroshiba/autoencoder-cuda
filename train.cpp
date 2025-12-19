#include "dataset.hpp"
#include "optimizer.hpp"
#include "loss.cuh"

#include "autoencoder.hpp"
#include "trainer.hpp"

#include "utils/logger.hpp"

int main() {
    LOG("Starting Autoencoder Training Pipeline");

    Config::load("config.yaml");

    Autoencoder<Device::GPU> model;
	model.build();

	Dataset dataset("dataset/cifar-10-batches-bin");
	Optimizer::SGD optimizer;

	Trainer trainer;
	trainer.fit(model, dataset, optimizer, Loss::MSE());
    LOG("Training completed");

    std::string checkpoint_path = "checkpoint/gpu_autoencoder.dat";
    model.save(checkpoint_path);
    LOG("Model saved to", checkpoint_path);
}