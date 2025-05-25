# sgemm profile

## 3 sgemm_thread_tile_coalesced_access_v3

`sgemm_thread_tile_coalesced_access_v3`相比于`sgemm_thread_tile_naive_v2`版本产生了负优化，原本是要优化`global memory`的访存合并问题，但是反而加重了访存不合并的问题。

## 4 sgemm_one_dim_tid_access_v4

`sgemm_one_dim_tid_access_v4`从global memory加载数据到shared memory时，将tid按一维平摊展开，与之相比v3是用了y,x两个维度的两层for循环。

## 5 sgemm_register_compress_v5

相比于v4, 调整了常量的声明位置，将寄存器使用从66个减少到了64个

## 