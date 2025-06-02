
project(cuda_opengl_demo LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.20)
  
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

 MESSAGE(STATUS "#########################")
MESSAGE(STATUS "${PROJECT_NAME}")
#MESSAGE(STATUS "#########################")

# Source file
if(${OpenGL_FOUND})
    if (${GLUT_FOUND})
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            message(STATUS "Will not build sample cuda_opengl_demo - not supported on aarch64")
        else()
            # Add target for cuda_opengl_demo
            add_executable(cuda_opengl_demo cuda_opengl_demo.cu cuda_opengl_demo.h main.cpp)

            target_compile_options(cuda_opengl_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

            target_compile_features(cuda_opengl_demo PRIVATE cxx_std_17 cuda_std_17)

            set_target_properties(cuda_opengl_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            target_include_directories(cuda_opengl_demo PUBLIC
                ${OPENGL_INCLUDE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
                ${GLUT_INCLUDE_DIRS}
            )

            target_link_libraries(cuda_opengl_demo
                ${OPENGL_LIBRARIES}
                ${GLUT_LIBRARIES}
            )
            if(WIN32)
                target_link_libraries(cuda_opengl_demo
                    ${PC_GLUT_LIBRARY_DIRS}/freeglut.lib
                    ${PC_GLUT_LIBRARY_DIRS}/glew64.lib
                )

                add_custom_command(TARGET cuda_opengl_demo
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy ${CMAKE_CURRENT_SOURCE_DIR}/../bin/win64/$<CONFIGURATION>/freeglut.dll
                    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
                )

                add_custom_command(TARGET cuda_opengl_demo
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E
                    copy ${CMAKE_CURRENT_SOURCE_DIR}/../bin/win64/$<CONFIGURATION>/glew64.dll
                    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>
                )
            endif()
        endif()
    else()
        message(STATUS "GLUT not found - will not build sample 'cuda_opengl_demo'")
    endif()
else()
    message(STATUS "OpenGL not found - will not build sample 'cuda_opengl_demo'")
endif()
