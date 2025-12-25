#include "autoencoder.hpp"
#include "dataset.hpp"
#include "config.hpp"
#include "utils/logger.hpp"
#include "utils/timer.cuh"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>

template <typename Tag>
int run_encoding(const std::string& model_path, const std::string& dataset_path) {

	// Load the pre-trained autoencoder model and dataset

	Autoencoder<Tag> model;

	try {
		model.load(model_path);
	} catch(const std::exception& e) {
		LOG("Error loading model:", e.what());
		return 1;
	}

	Dataset dataset(dataset_path);

	const size_t batch_size = 256;
	std::ofstream feat_out;
	std::ofstream label_out;

	// Training feature extraction

	size_t num_samples = dataset.train_size();
	LOG("Starting feature extraction on training set with ", num_samples, "samples");

	feat_out.open("content/output/train_features.bin", std::ios::binary);
	label_out.open("content/output/train_labels.bin", std::ios::binary);

	if(!feat_out || !label_out) {
		LOG("Failed to open output files");
		return 1;
	}

	size_t total_batches = (num_samples + batch_size - 1) / batch_size;
    float total_time = 0.0f;

	Timer timer;
	for(size_t i = 0; i < total_batches; ++i) {
		timer.start("batch");
		Tensor<Tag> batch;
		dataset.get_batch(i, batch_size, batch);
		
		Tensor<Tag> z_device = model.encode(batch);
		Tensor<Device::CPU> z_cpu = z_device;

		feat_out.write(reinterpret_cast<const char*>(z_cpu.data()), z_cpu.size() * sizeof(float));

		size_t current_batch_size = z_cpu.batches();
		size_t start_idx = i * batch_size;

		for(size_t j = 0; j < current_batch_size; ++j) {
			unsigned short lbl = dataset.train_label(start_idx + j);
			label_out.write(reinterpret_cast<const char*>(&lbl), sizeof(lbl));
		}

		timer.stop("batch");
		timer.synchronize();
		float batch_time = timer.elapsed("batch");
		total_time += batch_time;

		if((i + 1) % 10 == 0 || i == total_batches - 1)
			LOG("Processed batch", i + 1, "/", total_batches, "| Batch time:", batch_time, "ms");
	}

	feat_out.close();
	label_out.close();

	// Test feature extraction

	num_samples = dataset.test_size();
	LOG("Starting feature extraction on test set with ", num_samples, "samples");

	feat_out.open("content/output/test_features.bin", std::ios::binary);
	label_out.open("content/output/test_labels.bin", std::ios::binary);

	if(!feat_out || !label_out) {
		LOG("Failed to open output files for test set");
		return 1;
	}

	total_batches = (num_samples + batch_size - 1) / batch_size;

	for(size_t i = 0; i < total_batches; ++i) {
		timer.start("batch");
		Tensor<Tag> batch;
		dataset.get_test_batch(i, batch_size, batch);
		
		Tensor<Tag> z_device = model.encode(batch);
		Tensor<Device::CPU> z_cpu = z_device;

		feat_out.write(reinterpret_cast<const char*>(z_cpu.data()), z_cpu.size() * sizeof(float));

		size_t current_batch_size = z_cpu.batches();
		size_t start_idx = i * batch_size;

		for(size_t j = 0; j < current_batch_size; ++j) {
			unsigned short lbl = dataset.test_label(start_idx + j);
			label_out.write(reinterpret_cast<const char*>(&lbl), sizeof(lbl));
		}

		timer.stop("batch");
		timer.synchronize();
		float batch_time = timer.elapsed("batch");
		total_time += batch_time;

		if((i + 1) % 10 == 0 || i == total_batches - 1)
			LOG("Processed test batch", i + 1, "/", total_batches, "| Batch time:", batch_time, "ms");
	}

	LOG("Feature extraction done. Total time:", total_time / 1000.0f, "s");
	return 0;
}

int main(int argc, char** argv) {
	std::filesystem::create_directories("content/output");

	Config::load("config.yaml");

	std::string device = "gpu";
	if(argc > 1) device = argv[1];

	std::string model_path = "content/model/" + device + "_autoencoder.dat";
	if(argc > 2) model_path = argv[2];

	std::string dataset_path = "content/cifar-10-batches-bin";
	if(argc > 3) dataset_path = argv[3];

	if(device == "cpu") {
		LOG("Using CPU for encoding");
		return run_encoding<Device::CPU>(model_path, dataset_path);
	} else {
		LOG("Using GPU for encoding");
		return run_encoding<Device::GPU>(model_path, dataset_path);
	}
}
