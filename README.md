# ccuda_sample

CUDA   流处理器，核，线程块（threadblock），线程，网格（‌gridDim），块（block） 、 OpenGL 


# 一、CUDA基本命令

## 1、CUDA函数的定义
1.  __global__: 在GPU上定义，CPU上调用的函数
2. __device__: 在GPU上定义，由GPU调用的函数
3. __host__:   在CPU上定义的函数，一般与__device__合起来才会用到


## 2、CUDA上指针 cudaMalloc(**devPtr, byte_size)

## 3、GPU、CPU参数传递 cudaMemcopy(*dst, *src, byte_size, 类型)

类型 :

- CPU -> CPU   ===>  cudaMemcpyHostToHost
- CPU -> GPU   ===>  cudaMemcpyHostToDevice
- GPU -> CPU   ===>  cudaMemcpyDeviceToHost
- GPU -> GPU   ===>  cudaMemcpyDeviceToDevice 

## 4、核函数的调用

```
dim3  griddim(x, y, z);
dim3  blockdim(x, y, z);

function<<<griddim, blockdim>>>(参数);
```
