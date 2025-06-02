
project(02_cuda_texture LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.20)


MESSAGE(STATUS "#########################")
MESSAGE(STATUS "${PROJECT_NAME}")
#MESSAGE(STATUS "#########################")

 

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
 

INCLUDE_DIRECTORIES(
     
     D:/Work/ccuda/deps/OpenCV/include 
   # ${OpenCV_DIR}/include 
)

file(GLOB ALL_CXX_SRCS
	./kernel.cu
	)
# Source file
# Add target for cuda_streams
add_executable(${PROJECT_NAME} ${ALL_CXX_SRCS})
 
 

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)
# Á´½ÓOpenCV¿â
target_link_libraries(${PROJECT_NAME} D:/Work/ccuda/deps/OpenCV/x64/vc15/lib/opencv_world412d.lib)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Copy data files to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/cinque_terre_small.jpg
    ${CMAKE_CURRENT_BINARY_DIR}/
)

# Copy data files to output directory
#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#    ${CMAKE_CURRENT_SOURCE_DIR}/cinque_terre_small.jpg
#    ${CMAKE_CURRENT_BINARY_DIR}/
#)


add_custom_command(TARGET ${PROJECT_NAME}
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy D:/Work/ccuda/deps/OpenCV/x64/vc15/bin/opencv_world412d.dll
                    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
                )
