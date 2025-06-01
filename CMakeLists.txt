cmake_minimum_required(VERSION 3.20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

project(ccuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps)


message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
if(WIN32)
    set(PC_GLUT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps")
    set(PC_GLUT_LIBRARY_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/lib/x64")
endif()


find_package(OpenGL)
find_package(GLUT)


#set(CMAKE_PREFIX_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/../deps/OpenCV")
#
## 设置OpenCV_DIR变量
#set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../deps/OpenCV")
##set(OpenCV_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCV/x64/vc14/lib")
#
## 找到OpenCV库
#find_package(OpenCV REQUIRED)
#
## 添加OpenCV头文件目录
#include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
#add_executable(${PROJECT_NAME} main.cpp)

# 链接OpenCV库
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
# 检查 OpenCV 是否找到
#if (OpenCV_FOUND)
#    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
#else()
#    message(FATAL_ERROR "OpenCV not found.")
#endif()
#
## 将 OpenCV 的包含目录和库目录传递到主 CMakeLists.txt
#set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
#set(OPENCV_LIBS ${OpenCV_LIBS})
# 
#
#message(status " opendir ${OpenCV_DIR}")


INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/deps 
     ${PROJECT_SOURCE_DIR}/deps/OpenCV/include 
   # ${OpenCV_DIR}/include 
)



add_subdirectory (01_cuda_add)
add_subdirectory(02_cuda_texture)
add_subdirectory(cuda_streams)
add_subdirectory(cuda_opengl_demo)
