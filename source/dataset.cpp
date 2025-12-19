#include "dataset.hpp"
#include "config.hpp"
#include "utils/logger.hpp"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>

Dataset::Dataset(const std::string &path) {
	LOG("Initializing Dataset...");
	load(path);
	normalize();
	shuffle();
	LOG("Dataset ready. Train:", train_size(), "Test:", test_size());
}

void Dataset::load(const std::string &dataset_path) {
	constexpr size_t H = 32;
	constexpr size_t W = 32;
	constexpr size_t C = 3;
	constexpr size_t PIXELS = H * W;
	constexpr size_t ROW_BYTES = 1 + C * PIXELS;
	constexpr size_t TRAIN_N = 50000;
	constexpr size_t TEST_N = 10000;

	// Pre-allocate memory
	train_images = Tensor<Device::CPU>(TRAIN_N, C, H, W);
	train_labels.resize(TRAIN_N);
	
	test_images = Tensor<Device::CPU>(TEST_N, C, H, W);
	test_labels.resize(TEST_N);

	auto load_batch = [&](const std::vector<std::string> &files, Tensor<Device::CPU> &images, std::vector<unsigned short> &labels, size_t max) {
		std::vector<unsigned char> buffer(ROW_BYTES);
		size_t global_index = 0;

		for(const auto &file : files) {
			std::string full_path = dataset_path + "/" + file;
			std::ifstream in(full_path, std::ios::binary);
			
			if(!in) throw std::runtime_error("Failed to open: " + full_path);

			while(global_index < max && in.read(reinterpret_cast<char*>(buffer.data()), ROW_BYTES)) {
				labels[global_index] = static_cast<unsigned short>(buffer[0]);

				// Pointers for fast copy
				float* img_ptr = images.data() + (global_index * C * H * W);
				
				// CIFAR-10 binary is [Label, R-plane, G-plane, B-plane]
				// We convert uchar to float directly.
				// Note: Input is CHW, Tensor is CHW. We just need to strip the label.
				for(size_t i = 0; i < C * PIXELS; ++i) {
					img_ptr[i] = static_cast<float>(buffer[1 + i]);
				}
				global_index++;
			}
		}

		if(global_index != max) throw std::runtime_error("Dataset size mismatch");
	};

	const std::vector<std::string> train_files = {
		"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", 
		"data_batch_4.bin", "data_batch_5.bin"
	};
	LOG("Loading training data...");
	load_batch(train_files, train_images, train_labels, TRAIN_N);

	const std::vector<std::string> test_files = { "test_batch.bin" };
	LOG("Loading test data...");
	load_batch(test_files, test_images, test_labels, TEST_N);
}

void Dataset::normalize() {
	// Normalize [0, 255] -> [0.0, 1.0]
	auto norm_tensor = [](Tensor<Device::CPU> &t) {
		t.transform([](float x) { return x / 255.0f; });
	};

	norm_tensor(train_images);
	norm_tensor(test_images);
}

void Dataset::shuffle() {
	size_t n = train_images.batches();
	if(n == 0) return;

	std::vector<int> indices(n);
	std::iota(indices.begin(), indices.end(), 0);

	std::mt19937 rng(Config::seed);
	std::shuffle(indices.begin(), indices.end(), rng);

	Tensor<Device::CPU> shuffled_images(n, train_images.channels(), train_images.height(), train_images.width());
	std::vector<unsigned short> shuffled_labels(n);

	// Optimization: Move memory in blocks (strides) rather than pixel-by-pixel
	size_t stride = train_images.channels() * train_images.height() * train_images.width();
	
	const float* src_data = train_images.data();
	float* dst_data = shuffled_images.data();

	for(size_t i = 0; i < n; ++i) {
		int old_index = indices[i];
		shuffled_labels[i] = train_labels[old_index];

		// Block copy one image
		std::copy(
			src_data + (old_index * stride),
			src_data + ((old_index + 1) * stride),
			dst_data + (i * stride)
		);
	}

	train_images = std::move(shuffled_images);
	train_labels = std::move(shuffled_labels);
}

void Dataset::get_batch(size_t batch_index, size_t batch_size, Tensor<Device::CPU> &batch) {
	size_t total = train_images.batches();
	size_t start_index = batch_index * batch_size;
	
	if(start_index >= total) throw std::out_of_range("Batch index out of range");
	size_t actual_count = std::min(batch_size, total - start_index);
	
	// Resize batch tensor if necessary
	if(batch.batches() != actual_count ||
	   batch.channels() != train_images.channels() ||
	   batch.height() != train_images.height() ||
	   batch.width() != train_images.width()) {
		batch = Tensor<Device::CPU>(actual_count, train_images.channels(), train_images.height(), train_images.width());
	}

	// Optimization: Contiguous block copy
	size_t stride = train_images.channels() * train_images.height() * train_images.width();
	size_t total_elements = actual_count * stride;

	const float* src_ptr = train_images.data() + (start_index * stride);
	std::copy(src_ptr, src_ptr + total_elements, batch.data());
}

void Dataset::get_batch(size_t batch_index, size_t batch_size, Tensor<Device::GPU> &batch) {
	size_t total = train_images.batches();
	size_t start_index = batch_index * batch_size;
	
	if(start_index >= total) throw std::out_of_range("Batch index out of range");
	size_t actual_count = std::min(batch_size, total - start_index);
	
	// Resize batch tensor if necessary
	if(batch.batches() != actual_count ||
	   batch.channels() != train_images.channels() ||
	   batch.height() != train_images.height() ||
	   batch.width() != train_images.width()) {
		batch = Tensor<Device::GPU>(actual_count, train_images.channels(), train_images.height(), train_images.width());
	}

	// Optimization: Contiguous block copy
	size_t stride = train_images.channels() * train_images.height() * train_images.width();
	size_t total_elements = actual_count * stride;

	const float* src_ptr = train_images.data() + (start_index * stride);
	checkCUDA(cudaMemcpyAsync(batch.data(), src_ptr, total_elements * sizeof(float), cudaMemcpyHostToDevice, 0));
}

void Dataset::get_test_batch(size_t batch_index, size_t batch_size, Tensor<Device::CPU> &batch) {
	size_t total = test_images.batches();
	size_t start_index = batch_index * batch_size;
	
	if(start_index >= total) throw std::out_of_range("Batch index out of range");
	size_t actual_count = std::min(batch_size, total - start_index);
	
	// Resize batch tensor if necessary
	if(batch.batches() != actual_count ||
	   batch.channels() != test_images.channels() ||
	   batch.height() != test_images.height() ||
	   batch.width() != test_images.width()) {
		batch = Tensor<Device::CPU>(actual_count, test_images.channels(), test_images.height(), test_images.width());
	}

	// Optimization: Contiguous block copy
	size_t stride = test_images.channels() * test_images.height() * test_images.width();
	size_t total_elements = actual_count * stride;

	const float* src_ptr = test_images.data() + (start_index * stride);
	std::copy(src_ptr, src_ptr + total_elements, batch.data());
}

void Dataset::get_test_batch(size_t batch_index, size_t batch_size, Tensor<Device::GPU> &batch) {
	size_t total = test_images.batches();
	size_t start_index = batch_index * batch_size;
	
	if(start_index >= total) throw std::out_of_range("Batch index out of range");
	size_t actual_count = std::min(batch_size, total - start_index);
	
	// Resize batch tensor if necessary
	if(batch.batches() != actual_count ||
	   batch.channels() != test_images.channels() ||
	   batch.height() != test_images.height() ||
	   batch.width() != test_images.width()) {
		batch = Tensor<Device::GPU>(actual_count, test_images.channels(), test_images.height(), test_images.width());
	}

	// Optimization: Contiguous block copy
	size_t stride = test_images.channels() * test_images.height() * test_images.width();
	size_t total_elements = actual_count * stride;

	const float* src_ptr = test_images.data() + (start_index * stride);
	checkCUDA(cudaMemcpyAsync(batch.data(), src_ptr, total_elements * sizeof(float), cudaMemcpyHostToDevice, 0));
}

size_t Dataset::train_size() const { return train_images.batches(); }

size_t Dataset::test_size() const { return test_images.batches(); }

unsigned short Dataset::train_label(size_t index) const {
	if(index >= train_labels.size()) throw std::out_of_range("Label index out of bounds");
	return train_labels[index];
}