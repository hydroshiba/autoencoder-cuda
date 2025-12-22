cmake_minimum_required(VERSION 3.20)
project(autoencoder LANGUAGES CXX CUDA)

# C++ setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA setting
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add CUDA include directory for CXX files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Directories
include_directories(header template)
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

# Debug section

file(GLOB_RECURSE debsrc "source/tensor.cu" "source/tensor.cpp")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp")
	add_executable(debug ${debsrc} "debug.cpp")
	target_include_directories(debug PRIVATE header)
	target_compile_options(debug PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>: -O0 -g -lineinfo >
		$<$<COMPILE_LANGUAGE:CXX>: -O0 -g >
	)
endif()
