import torch
from torch.fx import symbolic_trace, replace_pattern


'''
How to Use the FX Subgraph Rewriter

For easy subgraph rewriting, FX exposes the utility function:

    replace_pattern(gm : GraphModule,
                    pattern : Callable,
                    replacement : Callable)
                    -> None

`replace_pattern` matches all possible non-overlapping sets of operators
and their data dependencies (`pattern`) in the Graph of a GraphModule
(`gm`), then replaces each of these matched subgraphs with another
subgraph (`replacement).

The docstring for `replace_pattern` (located in `subgraph_rewriter.py`)
gives an in-depth explanation as to how `pattern` and `replacement`
should be specified, what happens during pattern matching, and other
important technical details. This tutorial, therefore, is only meant to
give an overview as to the FX Subgraph Rewriter's basic functionality.
Let's go rewrite a Graph!
'''
# 如何使用 FX 子图重写器。
# 工具函数 replace_pattern(gm, pattern, replacement) 在 GraphModule 的图中查找
# 所有互不重叠、操作及数据依赖均匹配 pattern 的子图，并替换为 replacement。
# 有关模式定义、替换定义及匹配过程的详细说明见 subgraph_rewriter.py 中的函数文档。
# 本教程只概述其基础功能。

# Sample module
# 示例模块。
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = torch.neg(w1)
        m1 = torch.cat([val1, w2]).sum()
        val2 = torch.neg(w1)
        m2 = torch.cat([val2, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

# Symbolically trace an instance of `M`
# 对 M 的实例进行符号追踪。
traced = symbolic_trace(M())

# Define the pattern. The FX Subgraph Rewriter will match all
# 定义模式。FX 子图重写器会匹配
# non-overlapping instances of the pattern in the larger graph.
# 大图中所有互不重叠的模式实例。
# Note that Pattern-matching is done based on data dependencies,
# 注意，模式匹配依据数据依赖，
# not Node names. Even though we're operating on Nodes named `a1` and
# 而不是节点名称。即使操作的节点叫 a1、
# `a2` instead of `w1` and `w2`, the pattern is still a valid match
# a2，而不叫 w1、w2，这个模式仍能匹配
# for the two instances of `torch.cat([w1, w2]).sum()` above. Only
# 上面两处 torch.cat([w1, w2]).sum()。
# operations that contribute to the single output value of the pattern
# 只考虑对模式唯一输出值有贡献的
# are considered
# 操作。
def pattern(a1, a2):
    val1 = torch.neg(a1)
    return torch.cat([val1, a2]).sum()

# Define the replacement (same rules as the pattern)
# 定义替换模式，规则与匹配模式相同。
def replacement(w1, w2):
    return torch.stack([w1, w2])

# Replace `pattern` with `replacement` in `traced`
# 在 traced 中将 pattern 替换为 replacement。
replace_pattern(traced, pattern, replacement)

# After calling `replace_pattern`, the generated code is:
# 调用 replace_pattern 后，生成的代码如下：
'''
def forward(self, x, w1, w2):
    stack = torch.stack([w1, w2])
    max_1 = torch.max(stack);  stack = None
    add = x + max_1;  x = max_1 = None
    stack_1 = torch.stack([w1, w2]);  w1 = w2 = None
    max_2 = torch.max(stack_1);  stack_1 = None
    add_1 = add + max_2;  add = max_2 = None
    return add_1
'''
# 替换后的 forward 代码：对两组权重堆叠并取最大值，再分别与输入相加。
