# sgemm

## shared memory tiling

## register tiling

对shared memory可以进一步分块，把数据搬到寄存器中使用，减少shared memory的访问次数。

之前shared memory tiling中，每个block中处理一个`m * n`的小块，每个block中`m * n`个元素。

其中维护的shared memory有两块，smem_a：`m * k`，smem_b：`k * n`。

现在第一个问题，shared memory进一步分块之后，现在每个block需要多少个线程？

先假定shared memory的分块维度，将`m * n`的小块细分成`rm * rn`的小块，现在每个线程需要处理`rm * rn`小块的结果。

所以线程的数量减少为`(m + rm - 1) / rm * (n + rn - 1) / rn`。

那么线程加载从global memory加载数据到shared memory中也发生了变化，每个线程加载的数据更多了，但加载的逻辑并不需要修改。

