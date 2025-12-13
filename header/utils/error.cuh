#ifndef ERROR_CUH
#define ERROR_CUH

#include <iostream>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

namespace Error {

inline void checkCUDA(
	cudaError_t result,
	const char* file,
	int line,
	const char* func,
	bool should_throw = true)
{
	if(result != cudaSuccess) {
		std::string error_msg = "CUDA Runtime Error: " + std::string(cudaGetErrorString(result)) +
			" at " + std::string(file) + ":" + std::to_string(line) +
			" in function " + std::string(func);
		
		if(should_throw) throw std::runtime_error(error_msg);
		else {
			std::cerr << error_msg << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
}

}

#define checkCUDA(result, ...) Error::checkCUDA(result, __FILE__, __LINE__, __func__, ##__VA_ARGS__)

#endif // ERROR_CUH