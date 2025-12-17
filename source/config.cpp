#include "config.hpp"

size_t Config::batch_size = 64;
size_t Config::epochs = 20;
float Config::learning_rate = 1e-3;

size_t Config::Tensor::block_width = 16;
size_t Config::Tensor::block_height = 16;

size_t Config::Conv2D::block_width = 16;
size_t Config::Conv2D::block_height = 16;

size_t Config::MaxPool2D::block_width = 16;
size_t Config::MaxPool2D::block_height = 16;

size_t Config::Upsample2D::block_width = 16;
size_t Config::Upsample2D::block_height = 16;

#include <iostream>
#include <fstream>
#include <string>

#include <fkYAML/node.hpp>
#include "utils/logger.hpp"

void Config::load(const std::string& path) {
	std::ifstream fin(path);
	
	if(!fin) {
		LOG("Failed to open configuration file: ", path);
		LOG("Using default configuration values.");
		return;
	}

	fkyaml::node config = fkyaml::node::deserialize(fin);
	fin.close();

	// Load global configuration parameters

	try { batch_size = config.at("batch_size").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { epochs = config.at("epochs").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { learning_rate = config.at("learning_rate").get_value<float>(); }
	catch (const fkyaml::exception& exception) {}

	// Load class-specific configurations

	// Tensor
	try { Tensor::block_width = config["Tensor"].at("block_width").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { Tensor::block_height = config["Tensor"].at("block_height").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	// Conv2D
	try { Conv2D::block_width = config["Conv2D"].at("block_width").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { Conv2D::block_height = config["Conv2D"].at("block_height").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	// MaxPool2D
	try { MaxPool2D::block_width = config["MaxPool2D"].at("block_width").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { MaxPool2D::block_height = config["MaxPool2D"].at("block_height").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	// Upsample2D
	try { Upsample2D::block_width = config["Upsample2D"].at("block_width").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	try { Upsample2D::block_height = config["Upsample2D"].at("block_height").get_value<size_t>(); }
	catch (const fkyaml::exception& exception) {}

	// Log loaded configuration
	LOG("Configuration loaded from: ", path);

	LOG("Batch size:", batch_size);
	LOG("Epochs:", epochs);
	LOG("Learning rate:", learning_rate);

	LOG("Tensor block size:", std::to_string(Tensor::block_width) + "x" + std::to_string(Tensor::block_height));
	LOG("Conv2D block size:", std::to_string(Conv2D::block_width) + "x" + std::to_string(Conv2D::block_height));
	LOG("MaxPool2D block size:", std::to_string(MaxPool2D::block_width) + "x" + std::to_string(MaxPool2D::block_height));
	LOG("Upsample2D block size:", std::to_string(Upsample2D::block_width) + "x" + std::to_string(Upsample2D::block_height));
}