cmake_minimum_required(VERSION 3.20)
project(autoencoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(header)

file(GLOB_RECURSE sources "source/*.cpp" "source/*.cu")
add_executable(train ${sources} "train.cpp")
add_executable(encode ${sources} "encode.cpp")