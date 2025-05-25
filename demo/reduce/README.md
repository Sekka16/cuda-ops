# reduce优化

## block_all_reduce_f32

第一轮：

这里在读取数据的时候是直接从global memory到register，每个线程读取一个数据。

shared memory里面保存的是第一轮在warp内的规约结果，大小的block中warp的数量。

举例：现在block中256个线程，warp size是32，那么warp nums就等于8

第二轮：

这里读取数据的时候是shared memory里面读取，并且实际上只有第0个warp读取的数据会被用到。

**限制：warp nums必须小于等于32**

对第一轮shared memory中的结果再次进行规约，只在第0个warp中再做一次reduce。

最终每个block只有一个结果，这个结果通过atomicAdd加到全局内存中。

>> 所以这里会存在一个问题，如果block的数量很多，那么atomicAdd会比较低效。解决方法是每个block的结果存储到连续的内存上在进行一个block all reduce。