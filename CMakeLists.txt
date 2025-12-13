cmake_minimum_required(VERSION 3.20)
project(autoencoder LANGUAGES CXX CUDA)

# C++ setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA setting
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Directories
include_directories(header)
file(GLOB_RECURSE sources "source/*.cpp" "source/*.cu")

# Executables
add_executable(train ${sources} "train.cpp")
target_include_directories(train PRIVATE header)

add_executable(encode ${sources} "encode.cpp")
target_include_directories(encode PRIVATE header)

target_compile_options(train PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>: -O3 -lineinfo >
	$<$<COMPILE_LANGUAGE:CXX>: -O3 >
)

target_compile_options(encode PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>: -O3 -lineinfo >
	$<$<COMPILE_LANGUAGE:CXX>: -O3 >
)

# Debug executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp")
	add_executable(debug ${sources} "debug.cpp")
	target_include_directories(debug PRIVATE header)
	target_compile_options(debug PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>: -O0 -g -lineinfo >
		$<$<COMPILE_LANGUAGE:CXX>: -O0 -g >
	)
endif()