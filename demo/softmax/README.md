# softmax

## softmax naive one dim

一维度的softmax，可能出现的情况是，一个block无法处理所有的数据，但由于softmax的分母需要实现一个累加求和，那么必然涉及到各个block间的同步，以计算出累加和。（规约操作的跨block同步问题）

这其实是一个reduce的过程，reduce完之后在进行除法计算得到最终结果。

reduce累加所有block的局部和的过程可以用atomicAdd，block数量过大则需要两阶段的reduce，这里暂时只考虑atomicAdd。

