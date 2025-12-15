cmake_minimum_required(VERSION 3.20)
project(autoencoder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA and set standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile .cu files with separable compilation for device linking when needed
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# GPU MX450 = Turing
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(header)

file(GLOB_RECURSE sources "source/*.cpp" "source/*.cu")

add_executable(train ${sources} "train.cpp")
add_executable(encode ${sources} "encode.cpp")

# Ensure position-independent code and proper runtime for CUDA targets
foreach(tgt train encode)
	set_target_properties(${tgt} PROPERTIES
		POSITION_INDEPENDENT_CODE ON
	)
endforeach()