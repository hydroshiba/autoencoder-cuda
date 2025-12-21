cmake_minimum_required(VERSION 3.20)
project(autoencoder LANGUAGES CXX CUDA)
find_package(OpenMP REQUIRED)

# C++ setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA setting
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75) # Tesla T4

# Add CUDA include directory for CXX files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Directories
include_directories(lib header template)
file(GLOB_RECURSE sources "source/*.cpp" "source/*.cu")

# Force all files to be compiled by NVCC
set_source_files_properties(
	${sources}
	"train.cpp"
	"encode.cpp"
	"verify.cpp"
	"debug.cpp"
	PROPERTIES LANGUAGE CUDA
)

# Optimization flags
if(MSVC)
	# Windows/MSVC specific flags
	set(FLAGS 
		"-Xcompiler" "/O2"          # Optimize for speed
		"-Xcompiler" "/arch:AVX2"   # Use AVX2 instructions
		"-Xcompiler" "/fp:fast"     # Fast floating point math
		"-Xcompiler" "/openmp"      # Enable OpenMP
	)
else()
	# Linux/GCC/Clang specific flags
	set(FLAGS
		"-Xcompiler" "-O3"            # Optimize for speed
		"-Xcompiler" "-march=native"  # Compile for machine's specific CPU architecture
		"-Xcompiler" "-ffast-math"    # Fast floating point math (less precision, more speed)
		"-Xcompiler" "-fopenmp"       # Enable OpenMP
	)
endif()

# Compilation targets
add_executable(train ${sources} "train.cpp")
target_include_directories(train PRIVATE header)
target_link_libraries(train PRIVATE OpenMP::OpenMP_CXX)

add_executable(encode ${sources} "encode.cpp")
target_include_directories(encode PRIVATE header)
target_link_libraries(encode PRIVATE OpenMP::OpenMP_CXX)

add_executable(verify ${sources} "verify.cpp")
target_include_directories(verify PRIVATE header)
target_link_libraries(verify PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(train PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>: 
		-O3 
		-lineinfo 
		--extended-lambda 
		--expt-relaxed-constexpr 
		${FLAGS} 
	>
)

target_compile_options(encode PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>: 
		-O3 
		-lineinfo 
		--extended-lambda 
		--expt-relaxed-constexpr 
		${FLAGS} 
	>
)

target_compile_options(verify PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>: 
		-O3 
		-lineinfo 
		--extended-lambda 
		--expt-relaxed-constexpr 
		${FLAGS} 
	>
)

# Debug section
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp")
	add_executable(debug ${sources} "debug.cpp")
	target_include_directories(debug PRIVATE header)
	target_link_libraries(debug PRIVATE OpenMP::OpenMP_CXX)
	target_compile_options(debug PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>: -lineinfo --extended-lambda --expt-relaxed-constexpr ${FLAGS} >
	)
endif()