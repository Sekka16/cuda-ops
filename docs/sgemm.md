# sgemm

$$
A：M * K \\
B: K * N \\
C: M * N \\
$$

## shared memory tiling

将原矩阵分成小块读入共享内存，每一个block计算出一个`m * n`的结果。

为此需要准备两块共享内存，共享内存的大小由程序员自行控制。

$$
sa: m * k \\
sb: k * n 
$$

每个block中需要`m * n`个线程，每个线程计算出一个结果。

每个线程共`(K + k - 1) / k`轮迭代累加出最终的结果。

第一个问题：block中的线程需要读取`k (m + n)`个元素，怎么分配给`m * n`个线程？

每个线程读取一个元素是有问题的，例如读取smem_a共`m * k`个元素，而共有`m * n`个线程，如果`n < k`就会有元素没有线程读取。

为了解决这个问题，需要一个循环，在每轮循环中，block中的`m * n`个线程读取了`m * n`个元素，下一轮循环，循环变量也应该增加`m * n`，然后再计算其在`m * k`的共享内存中的index和全局中的index。

```cuda
int num_a_elements = SRAM_M * SRAM_K;
for (int i = tid; i < num_a_elements; i += threads_per_block) {
    int a_row = i / SRAM_K;
    int a_col = i % SRAM_K;
    int global_row = by * SRAM_M + a_row;
    int global_col = k0 + a_col;
    smem_a[a_row][a_col] = (global_row < M && global_col < K) ? a[global_row * K + global_col] : 0.0f;
}
```

## register tiling

对shared memory可以进一步分块，把数据搬到寄存器中使用，减少shared memory的访问次数。

之前shared memory tiling中，每个block中处理一个`m * n`的小块，每个block中`m * n`个元素。

其中维护的shared memory有两块，smem_a：`m * k`，smem_b：`k * n`。

现在第一个问题，shared memory进一步分块之后，现在每个block需要多少个线程？

先假定shared memory的分块维度，将`m * n`的小块细分成`rm * rn`的小块，现在每个线程需要处理`rm * rn`小块的结果。

所以线程的数量减少为`(m + rm - 1) / rm * (n + rn - 1) / rn`。

那么线程加载从global memory加载数据到shared memory中也发生了变化，每个线程加载的数据更多了，但加载的逻辑并不需要修改。

