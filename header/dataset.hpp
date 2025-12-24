#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

#include "tensor.hpp"
#include "utils/device.hpp"

class Dataset {
private:
	Tensor<Device::CPU> train_images;
	std::vector<unsigned short> train_labels;

	Tensor<Device::CPU> test_images;
	std::vector<unsigned short> test_labels;

	void load(const std::string &path);
	void normalize();
	void shuffle(); 

public:
	Dataset(const std::string &path);

	// Methods that return a data batch
	void get_batch(size_t batch_index, size_t batch_size, Tensor<Device::CPU> &batch);
	void get_batch(size_t batch_index, size_t batch_size, Tensor<Device::GPU> &batch);

	void get_test_batch(size_t batch_index, size_t batch_size, Tensor<Device::CPU> &batch);
	void get_test_batch(size_t batch_index, size_t batch_size, Tensor<Device::GPU> &batch);

	template <typename Tag>
	Tensor<Tag> get_batch(size_t batch_index, size_t batch_size) {
		Tensor<Tag> batch;
		get_batch(batch_index, batch_size, batch);
		return batch;
	}

	template <typename Tag>
	Tensor<Tag> get_test_batch(size_t batch_index, size_t batch_size) {
		Tensor<Tag> batch;
		get_test_batch(batch_index, batch_size, batch);
		return batch;
	}
	
	// Getters
	size_t test_size() const;
	size_t train_size() const;
	unsigned short train_label(size_t index) const;
};

#endif // DATASET_HPP