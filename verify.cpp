#include <iostream>
#include <fstream>
#include <filesystem>

#include "dataset.hpp"
#include "autoencoder.hpp"
#include "loss.cuh"
#include "config.hpp"

#include "utils/random.cuh"
#include "utils/logger.hpp"
#include "utils/kernel.cuh"

std::vector<unsigned char> to_PPM(const Tensor<Device::CPU> &tensor) {
	const int N = tensor.batches();
	const int C = tensor.channels();
	const int H = tensor.height();
	const int W = tensor.width();

	if(N != 1 || C != 3)
		throw std::invalid_argument("to_PPM: Tensor must have shape (1, 3, H, W).");

	std::vector<unsigned char> ppm_data(H * W * 3);

	for(int h = 0; h < H; ++h) {
		for(int w = 0; w < W; ++w) {
			for(int c = 0; c < 3; ++c) {
				// Tensor is in CHW format, PPM is HWC
				size_t tensor_idx = (c * H * W) + (h * W + w);
				size_t ppm_idx = (h * W + w) * 3 + c;
				
				float val = tensor.data()[tensor_idx];
				ppm_data[ppm_idx] = static_cast<unsigned char>(std::min(std::max(val * 255.0f, 0.0f), 255.0f));
			}
		}
	}

	return ppm_data;
}

int main(int argc, char** argv) {
	LOG("Starting model verification");
	Config::load("config.yaml");

	std::string model_path = argc > 1 ? argv[1] : "content/model/model.dat";
	LOG("Using model file:", model_path);
	std::string dataset_path = argc > 2 ? argv[2] : "content/cifar-10-batches-bin";
	LOG("Using dataset path:", dataset_path);

	// Get an image to test
	Dataset dataset(dataset_path);
	size_t seed = Config::seed;
	size_t image_index = Random::splitmix64(seed) % dataset.test_size();
	LOG("Using image index:", image_index, "in testing set");
	
	Tensor<Device::CPU> cpu_batch;
	Tensor<Device::GPU> gpu_batch;
	dataset.get_test_batch(image_index, 1, cpu_batch);
	dataset.get_test_batch(image_index, 1, gpu_batch);

	// Load models
	Autoencoder<Device::CPU> cpu_model;
	Autoencoder<Device::GPU> gpu_model;

	cpu_model.load(model_path);
	gpu_model.load(model_path);

	// Loss sanity check
	Loss::MSE mse;
	float batch_diff = mse(cpu_batch, Tensor<Device::CPU>(gpu_batch));
	LOG("Initial CPU-GPU batch difference (should be 0):", batch_diff);

	// Infer the models and compare outputs
	Tensor<Device::CPU> cpu_output = cpu_model.forward(cpu_batch);
	Tensor<Device::GPU> gpu_output = gpu_model.forward(gpu_batch);

	// Compare outputs
	float cpu_loss = mse(cpu_output, cpu_batch);
	float gpu_loss = mse(gpu_output, gpu_batch);

	LOG("CPU Loss:", cpu_loss);
	LOG("GPU Loss:", gpu_loss);

	Tensor<Device::CPU> diff = gpu_output - cpu_output;

	float max_diff = 0.0f;
	float mae = 0.0f;

	for(size_t i = 0; i < diff.size(); ++i) {
		float val = std::abs(diff.data()[i]);
		mae += val;
		if(val > max_diff) max_diff = val;
	}

	mae /= diff.size();

	LOG("Verification Results:");
	LOG("Max Absolute Difference:", max_diff);
	LOG("Mean Absolute Error:", mae);

	// Save output images
	std::string directory = "content/output/";
	std::filesystem::create_directory(directory);

	auto in_ppm = to_PPM(cpu_batch);
	auto cpu_ppm = to_PPM(cpu_output);
	auto gpu_ppm = to_PPM(Tensor<Device::CPU>(gpu_output));

	std::ofstream in_file(directory + "in.ppm", std::ios::binary);
	std::ofstream cpu_file(directory + "cpu.ppm", std::ios::binary);
	std::ofstream gpu_file(directory + "gpu.ppm", std::ios::binary);

	const std::string ppm_header = "P6\n32 32\n255\n";
	in_file << ppm_header;
	cpu_file << ppm_header;
	gpu_file << ppm_header;

	in_file.write(reinterpret_cast<const char*>(in_ppm.data()), in_ppm.size());
	cpu_file.write(reinterpret_cast<const char*>(cpu_ppm.data()), cpu_ppm.size());
	gpu_file.write(reinterpret_cast<const char*>(gpu_ppm.data()), gpu_ppm.size());
}