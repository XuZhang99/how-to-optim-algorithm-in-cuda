> 博客来源：https://leimao.github.io/blog/CUDA-Data-Alignment/ ，来自Lei Mao，已获得作者转载授权。

# CUDA 数据对齐

## 简介

为了获得最佳性能，与C++中的数据对齐要求类似(https://leimao.github.io/blog/CPP-Data-Alignment/)，CUDA也需要数据对齐。

在这篇博客文章中，我将快速讨论CUDA中的数据对齐要求。

## 全局内存的合并访问

全局内存驻留在设备内存中，设备内存通过32、64或128字节的内存事务进行访问。这些内存事务必须自然对齐：只有对齐到其大小的32、64或128字节的设备内存段（即，其首地址是其大小的倍数）才能被内存事务读取或写入。

当一个线程束执行访问全局内存的指令时，它会根据每个线程访问的字大小和内存地址在线程间的分布，将线程束内线程的内存访问合并为一个或多个内存事务。一般来说，需要的事务越多，除了线程访问的字之外传输的未使用字就越多，相应地降低了指令吞吐量。

对于计算能力6.0或更高的设备，要求可以很容易地总结：线程束中线程的并发访问将合并为多个事务，事务数量等于服务线程束中所有线程所需的32字节事务数量。

驻留在全局内存中的变量地址或驱动程序或运行时API的内存分配例程（如`cudaMalloc`或`cudaMallocPitch`）返回的任何地址始终至少对齐到256字节。

## 示例

例如，如果由32个线程组成的线程束中的每个线程都想读取4字节数据，并且如果线程束中所有线程的4字节数据（128字节数据）彼此相邻且32字节对齐，即第一个4字节数据的地址是32的倍数，那么内存访问是合并的，GPU将进行$\frac{4\times 32}{32}=4$次32字节内存事务。由于GPU进行了尽可能少的事务，实现了最大内存事务吞吐量。

如果128字节数据在内存上不是32字节对齐的，比如说是4字节对齐的，那么将必须进行一次额外的32字节内存事务，因此内存访问吞吐量变为最大理论吞吐量的$\frac{4}{5}=80%$。（跨了5个数据段）

此外，如果所有线程的4字节数据彼此不相邻并且在内存上稀疏分散，那么可能需要进行最多32次32字节内存事务，吞吐量仅为最大理论吞吐量的$\frac{4}{32}=12.5%$。

## 大小和对齐要求

全局内存指令支持读取或写入大小等于1、2、4、8或16字节的字。只有当数据类型的大小为1、2、4、8或16字节且数据自然对齐（即，其地址是该大小的倍数）时，对驻留在全局内存中的数据的任何访问（通过变量或指针）才会编译为单个全局内存指令。

如果不满足此大小和对齐要求，访问将编译为多个具有交错访问模式的指令，这些指令阻止这些指令完全合并。因此，建议对驻留在全局内存中的数据使用满足此要求的类型。

读取非自然对齐的8字节或16字节字会产生错误结果（偏差几个字），因此必须特别注意维护这些类型的任何值或值数组的起始地址的对齐。

因此，使用大小等于1、2、4、8或16字节的字有时很简单，因为如上所述，内存分配CUDA API返回的起始内存地址始终至少对齐到256字节，这已经是1、2、4、8或16字节对齐的。所以我们可以安全地将字序列（如数值数组、矩阵或张量）保存到分配的内存中，而不必担心读取8字节或16字节大小的字会产生错误结果。为了实现最佳的内存访问吞吐量，在kernel实现上需要特别注意，以便合并的内存访问也是自然对齐的。

但是，如果字大小不是1、2、4、8或16字节怎么办？内存分配CUDA API返回的起始内存地址将不保证它是自然对齐的，因此内存访问吞吐量将显著受损。通常有两种方法来处理这个问题。

- 使用内置向量类型(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#built-in-vector-types)，其对齐要求已经指定并满足。
- 类似于GCC中用于强制结构数据对齐的编译器指定符`alignas`，在NVCC中使用编译器指定符`__align__`来强制结构数据对齐。

```c++
struct __align__(4) int8_3_4_t
{
    int8_t x;
    int8_t y;
    int8_t z;
};

struct __align__(16) float3_16_t
{
    float x;
    float y;
    float z;
};
```

## 结论

始终使字大小等于1、2、4、8或16字节，并使数据自然对齐。

如果分配的内存仅用于相同类型字的序列，读取字并产生错误结果的情况很少发生，因为内存分配CUDA API返回的起始内存地址始终至少对齐到256字节。但是，如果为不同类型的多个字序列分配一大块内存（带有或不带有填充），必须特别注意维护任何字或字序列的起始地址的对齐，因为它可能产生错误结果（对于非自然对齐的8字节或16字节字）。

## 参考文献

- C++ Data Alignment(https://leimao.github.io/blog/CPP-Data-Alignment/)
- CUDA Device Memory Access(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-programming-guide/index.html#device-memory-accesses)
- Coalesced Access to Global Memory(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)


