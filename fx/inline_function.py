import torch
from torch.fx import Proxy, symbolic_trace
from torch.fx.node import map_arg


'''
How to Inline a Function Into an Existing Graph

One reason you might want to inline a function is to get around FX's
default tracing behavior. For example, unless you've defined a custom
Tracer, the out-of-the-box implementation of ``symbolic_trace`` causes
references to ``torch.nn`` module instances to appear as
``call_module`` calls rather than being traced through. Let's say this
behavior is almost what you need; the only problem is that there's a
single module call that you want to replace with an inlined trace of the
function. Creating a custom Tracer would be too much. Instead, you can
accomplish this using Proxies.

The following code demonstrates how to trace a module and inline it
into an existing Graph using Proxy. We'll trace our Graph, then iterate
through its Nodes until we find the right place to swap out the
``call_module`` Node with an inlined trace. At that point, we'll create
Proxies from the Node's args and kwargs. Finally, we'll call the
function we want to replace with those Proxies--which will, in essence,
"trace" that function. Finally, we'll insert the result of that call
into our Graph. (This last step will automatically inline the function.)
'''
# 如何将函数内联到现有计算图中。
# FX 默认将 torch.nn 模块调用保留为 call_module，而不展开内部实现。
# 如果只需将一个模块调用展开，无需另建追踪器，可以使用 Proxy。
# 先追踪模块，再遍历节点，找到待替换的 call_module；基于它的 args、kwargs 创建 Proxy，
# 将这些 Proxy 传给目标函数，即可追踪其内部实现。最后把调用结果插入现有图，完成内联。


# Sample module
# 示例模块。
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x) + 1.0

# Symbolically trace an instance of `M`. After tracing, `self.relu` is
# 对 M 的实例进行符号追踪。追踪后，self.relu
# represented as a `call_module` Node. The full operation in the
# 表示为 call_module 节点；生成的 forward 函数中
# generated `forward` function's code will appear as `self.relu(x)`
# 对应的完整操作显示为 self.relu(x)。
m = symbolic_trace(M())

# Insert nodes from the ReLU graph in place of the original call to
# 用 ReLU 计算图中的节点替换原来的调用节点。
# `self.relu`
# create a graph-appending tracer pointing to the original graph
# 创建一个指向原图、用于向图中追加节点的追踪器。
tracer = torch.fx.proxy.GraphAppendingTracer(m.graph)
for node in m.graph.nodes:
    # Find `call_module` Node in `m` that corresponds to `self.relu`.
    # 在 m 中找到与 self.relu 对应的 call_module 节点。
    # This is the Node we want to swap out for an inlined version of the
    # 要将这个节点替换为同一调用的
    # same call
    # 内联版本。
    if (node.op, node.target) == ("call_module", "relu"):
        with m.graph.inserting_before(node):
            # Create a Proxy from each Node in the current Node's
            # 将当前节点 args 和 kwargs 中的每个节点
            # args/kwargs
            # 转换为 Proxy。
            proxy_args = map_arg(node.args, lambda n: Proxy(n, tracer))
            proxy_kwargs = map_arg(node.kwargs, lambda n: Proxy(n, tracer))
            # Call `m.relu` with the newly-created Proxy arguments.
            # 使用新建的 Proxy 参数调用 m.relu。
            # `m.relu` is the generic version of the function; by
            # m.relu 是该函数的通用版本；
            # calling it with Proxies created from Nodes in `m`, we're
            # 传入由 m 中节点创建的 Proxy，
            # emitting Nodes that reference exiting values in the IR.
            # 就会生成引用中间表示（IR）中既有值的新节点。
            # The result of this call is another Proxy, which we can
            # 调用结果是另一个 Proxy，可将其
            # hook into our existing Graph to complete the function
            # 接入现有计算图，从而完成函数的
            # inlining.
            # 内联展开。
            proxy_output = m.relu(*proxy_args, **proxy_kwargs)
            # Replace the relu `call_module` node with the inlined
            # 将 relu 的 call_module 节点替换为
            # version of the function
            # 函数的内联版本。
            node.replace_all_uses_with(proxy_output.node)
            # Make sure that the old relu Node is erased
            # 确保删除旧的 relu 节点。
            m.graph.erase_node(node)
