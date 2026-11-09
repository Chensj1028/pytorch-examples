import torch
from torch.fx import symbolic_trace
import operator

"""
How to Replace One Op With Another

1. Iterate through all Nodes in your GraphModule's Graph.
2. Determine if the current Node should be replaced. (Suggested: match
on the Node's ``target`` attribute).
3. Create a replacement Node and add it to the Graph.
4. Use the FX built-in ``replace_all_uses_with`` to replace all uses of
the current Node with the replacement.
5. Delete the old Node from the graph.
6. Call ``recompile`` on the GraphModule. This updates the generated
Python code to reflect the new Graph state.

Currently, FX does not provide any way to guarantee that replaced
operators are syntactically valid. It's up to the user to confirm that
any new operators will work with the existing operands.

The following code demonstrates an example of replacing any instance of
addition with a bitwise AND.

To examine how the Graph evolves during op replacement, add the
statement `print(traced.graph)` after the line you want to inspect.
Alternatively, call `traced.graph.print_tabular()` to see the IR in a
tabular format.
"""
# 如何将一种操作替换为另一种：
# 1．遍历 GraphModule 中的所有节点；2．判断是否需要替换，建议匹配 target 属性；
# 3．创建替代节点并加入图；4．用 replace_all_uses_with 替换原节点的所有使用位置；
# 5．删除旧节点；6．调用 recompile，使生成的 Python 代码反映新图。
# FX 不保证替换后的操作与操作数兼容，需要调用方自行确认。
# 本例将加法替换为按位与；可插入 print(traced.graph) 或调用 print_tabular() 查看图的变化。

# Sample module
# 示例模块。
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)

# Symbolically trace an instance of the module
# 对模块实例进行符号追踪。
traced = symbolic_trace(M())

# As demonstrated in the above example, there are several different ways
# 如上例所示，加法可以通过几种不同方式
# to denote addition. The possible cases are:
# 表示，具体情况如下：
#     1. `x + y` - A `call_function` Node with target `operator.add`.
# 1．x + y：目标为 operator.add 的 call_function 节点。
#         We can match for equality on that `operator.add` directly.
# 可直接比较目标是否等于 operator.add。
#     2. `torch.add(x, y)` - A `call_function` Node with target
# 2．torch.add(x, y)：目标为 torch.add 的
#         `torch.add`. Similarly, we can match this function directly.
# call_function 节点，同样可直接匹配该函数。
#     3. `x.add(y)` - The Tensor method call, whose target we can match
# 3．x.add(y)：张量方法调用，可通过目标名称
#         as a string.
# 字符串进行匹配。

patterns = set([operator.add, torch.add, "add"])

# Go through all the nodes in the Graph
# 遍历计算图中的所有节点。
for n in traced.graph.nodes:
    # If the target matches one of the patterns
    # 如果目标匹配某种模式，
    if any(n.target == pattern for pattern in patterns):
        # Set the insert point, add the new node, and replace all uses
        # 设置插入位置，添加新节点，并将 n 的所有使用位置
        # of `n` with the new node
        # 替换为新节点。
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        # 从计算图删除旧节点。
        traced.graph.erase_node(n)

# Don't forget to recompile!
# 别忘了重新编译。
traced.recompile()
