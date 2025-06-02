project(07_cuda_device_query LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.20)


 MESSAGE(STATUS "#########################")
MESSAGE(STATUS "${PROJECT_NAME}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


file(GLOB ALL_CXX_SRCS
	./main.cpp
	./README.md
	)
# Source file
# Add target for cuda_streams
add_executable(${PROJECT_NAME} ${ALL_CXX_SRCS})

#target_sources(${PROJECT_NAME} PRIVATE README.md):ml-citation{ref="7" data="citationList"} 

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	 


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    CUDA::cudart
)