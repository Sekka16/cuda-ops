# sgemm profile

## 3 sgemm_thread_tile_coalesced_access_v3

`sgemm_thread_tile_coalesced_access_v3`相比于`sgemm_thread_tile_naive_v2`版本产生了负优化，原本是要优化`global memory`的访存合并问题，但是反而加重了访存不合并的问题。