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
	Tensor<Device::CPU> get_batch(size_t batch_index, size_t batch_size); // Returns a subset of the training data
	
	// Getters
	size_t train_size() const;
	size_t test_size() const;
	unsigned short train_label(size_t index) const;
};

#endif // DATASET_HPP