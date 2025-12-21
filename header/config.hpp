#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>

class Config {
public:
	// Prevent instantiation

	Config() = delete;
	Config(const Config&) = delete;
	Config& operator=(const Config&) = delete;

	// Global configuration parameters

	static size_t batch_size;
	static size_t epochs;
	static float learning_rate;
	static int seed;
	static int checkpoint_interval;

	// Class-specific configurations

	struct Tensor {
		Tensor() = delete;
		static size_t block_width;
		static size_t block_height;
	};

	struct Conv2D {
		Conv2D() = delete;
		static size_t block_width;
		static size_t block_height;
	};

	struct MaxPool2D {
		MaxPool2D() = delete;
		static size_t block_width;
		static size_t block_height;
	};

	struct Upsample2D {
		Upsample2D() = delete;
		static size_t block_width;
		static size_t block_height;
	};

	static void load(const std::string& path);
};

#endif // CONFIG_HPP