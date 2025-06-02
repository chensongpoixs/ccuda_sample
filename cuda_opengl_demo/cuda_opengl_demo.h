#ifndef CUDA_OPENGL_DEMO_H_
#define CUDA_OPENGL_DEMO_H_
#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void launch_kernel(float4* pos, unsigned int mesh_width,
    unsigned int mesh_height, float time);


#endif // CUDA_OPENGL_DEMO_H_