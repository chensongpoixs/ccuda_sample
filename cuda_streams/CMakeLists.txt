project(cuda_streams LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.20)


 MESSAGE(STATUS "#########################")
MESSAGE(STATUS "${PROJECT_NAME}")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)


file(GLOB ALL_CXX_SRCS
	./cuda_streams.cu
	)
# Source file
# Add target for cuda_streams
add_executable(${PROJECT_NAME} ${ALL_CXX_SRCS})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
