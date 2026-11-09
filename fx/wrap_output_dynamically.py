
from enum import Enum, auto

import torch
from torch.fx import GraphModule, Node, Proxy, symbolic_trace

'''
Wrap Graph Output Dynamically

The following code demonstrates how change an existing Graph based on
parameters specified at runtime. We'll let the user specify an
activation function from a predefined Enum list, then we'll symbolically
trace it. Next, we'll create a Proxy from the last operation in the
Graph. We'll call our traced activation function with this Proxy and
insert the ``output`` Node from that call into our Graph. (This final
step will automatically inline the entire traced function.)
'''
# 动态包装计算图输出。
# 根据运行时参数修改已有图：用户从预定义枚举中选择激活函数，随后对该函数进行符号追踪。
# 基于图中最后一个操作创建 Proxy，用它调用追踪后的激活函数，并将输出节点插入原图，
# 从而自动内联整个追踪函数。


# Sample module
# 示例模块。
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = torch.cat([x, y])
        return y

# Symbolically trace an instance of `M`
# 对 M 的实例进行符号追踪。
traced = symbolic_trace(M())

# Selected activation functions
# 可选激活函数。
class ActivationFunction(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    PRELU = auto()

# Map activation function names to their implementation
# 将激活函数名映射到具体实现。
activation_functions = {
    ActivationFunction.RELU: torch.nn.ReLU(),
    ActivationFunction.LEAKY_RELU: torch.nn.LeakyReLU(),
    ActivationFunction.PRELU: torch.nn.PReLU(),
}

def wrap_in_activation_function(m: GraphModule, fn: ActivationFunction) -> GraphModule:
    # Get output node
    # 获取输出节点。
    output_node: Optional[Node] = None
    for n in reversed(m.graph.nodes):
        if n.op == "output":
            output_node = n
            break
    assert output_node

    # Get the actual output (the "input" of the output node). This is
    # 获取实际输出，即输出节点的输入。
    # the Node we want to wrap in a user-specified activation function
    # 需要将此节点包装到用户指定的激活函数中。
    assert len(output_node.all_input_nodes) == 1
    wrap_node = output_node.all_input_nodes[0]

    # Wrap the actual output in a Proxy
    # 将实际输出包装为 Proxy。
    wrap_proxy = Proxy(wrap_node)

    # Get the implementation of the specified activation function and
    # 获取指定激活函数的实现，并对其
    # symbolically trace it
    # 进行符号追踪。
    fn_impl = activation_functions[fn]
    fn_impl_traced = symbolic_trace(fn_impl)

    # Call the specified activation function using the Proxy wrapper for
    # 使用 output_op 的 Proxy 包装对象
    # `output_op`. The result of this call is another Proxy, which we
    # 调用指定激活函数。返回结果也是一个 Proxy，
    # can hook into our existing Graph.
    # 可将其接入现有计算图。
    with traced.graph.inserting_after(wrap_node):
        fn_impl_output_node = fn_impl_traced(wrap_proxy)
        new_args = (fn_impl_output_node.node,)
        output_node.args = new_args

    m.recompile()


# Example call
# 调用示例。
x, y = torch.randn(5, 3), torch.randn(5, 3)
orig_output = traced(x, y)

wrap_in_activation_function(traced, ActivationFunction.LEAKY_RELU)
new_output = traced(x, y)

torch.testing.assert_close(new_output, torch.nn.LeakyReLU()(orig_output))
