 

// Utilities and system includes

#include <helper_cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinates
    // 计算u v 坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    // 计算简单正弦波模式
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    // write output vertex
    pos[y * width + x] = make_float4(u, w, v, 1.0f);
}


 void launch_kernel(float4* pos, unsigned int mesh_width,
    unsigned int mesh_height, float time)
{
    // execute the kernel  线程块尺寸
    dim3 block(8, 8, 1);
    // 网格尺寸
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    // <<<线程块， 多少个线程>>>
    simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
}


