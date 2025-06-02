#  一  CUDA代码的高效策略

1. 高效公式
1. 合并全局内存
1. 避免线程发散


# 二 Kernel加载方式

1. 查询本机参数
1. Kernel加载的1D/2D/3D模式
1. kernel函数的关键字


# 三 CUDA中的各种内存的代码使用

1. 全局内存
1. 共享内存
1. 本地内存

# 四 CUDA同步操作

1. 原子操作
1. 同步函数
1. CPU/GPU同步


__syncthreads() 函数只是在线程块内线程同步， 保证线程块内所有线程都执行到统一位置

# 五  并行化高效策略

1. 归约
1. 扫描



在CUDA编程中，blockSize和gridSize是用于定义核函数执行配置的关键参数13：

‌blockSize（线程块尺寸）‌

类型为dim3，表示每个线程块包含的线程数量13

在示例中dim3 blockSize(thread, thread)创建了二维线程块，每块包含thread×thread个线程5

每个线程块最大线程数限制为102457

同一线程块内的线程可通过共享内存通信47

‌gridSize（网格尺寸）‌

类型为dim3，表示网格中包含的线程块数量13

在示例中dim3 gridSize(grid)创建了一维网格，包含grid个线程块3

最大网格维度为65535（x/y/z方向）37

‌执行配置<<<gridSize, blockSize>>>‌

该语法指定核函数启动时的并行执行结构35

总线程数 = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z13

在rgba_to_greyscale示例中，表示使用grid个线程块，每个块有thread×thread个线程并行处理图像数据68

典型应用场景中，开发者需要根据数据规模和硬件特性调整这两个参数以获得最佳性能14。例如处理二维图像时，常使用二维线程块组织（如16×16）来匹配像素矩阵结构




CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce MX150"
  CUDA Driver Version / Runtime Version          12.9 / 12.9
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 2048 MBytes (2147352576 bytes)
MapSMtoCores for SM 6.1 is undefined.  Default to use 192 Cores/SM
MapSMtoCores for SM 6.1 is undefined.  Default to use 192 Cores/SM
  ( 3) Multiprocessors x (192) CUDA Cores/MP:    576 CUDA Cores
  GPU Clock rate:                                1532 MHz (1.53 GHz)
  Memory Clock rate:                             3004 Mhz
  Memory Bus Width:                              64-bit
  L2 Cache Size:                                 524288 bytes
  Max Texture Dimension Size (x,y,z)             1D=(131072), 2D=(131072,65536), 3D=(16384,16384,16384)
  Max Layered Texture Size (dim) x layers        1D=(32768) x 2048, 2D=(32768,32768) x 2048
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 5 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Bus ID / PCI location ID:           2 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.9, CUDA Runtime Version = 12.9, NumDevs = 1, Device0 = NVIDIA GeForce MX150





uint3 __device_builtin__ __STORAGE__ threadIdx;
uint3 __device_builtin__ __STORAGE__ blockIdx;
dim3 __device_builtin__ __STORAGE__ blockDim;
dim3 __device_builtin__ __STORAGE__ gridDim;
int __device_builtin__ __STORAGE__ warpSize;