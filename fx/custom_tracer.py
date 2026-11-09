import torch
from torch.fx import symbolic_trace, Tracer, Graph, GraphModule, Node
from typing import Any, Callable, Dict, Optional, Tuple, Union



"""
How to Create and Use Custom Tracers

`Tracer`--the class that implements the symbolic tracing functionality
of `torch.fx.symbolic_trace`--can be subclassed to override various
behaviors of the tracing process. In this tutorial, we'll demonstrate
how to customize the symbolic tracing process using some handwritten
Tracers. Each example will show that, by simply overriding a few methods
in the `Tracer` class, you can alter the Graph produced by symbolic
tracing. For a complete description of the methods that can be changed,
refer to the docstrings of the methods in the Tracer class. Information
can be found at: https://pytorch.org/docs/master/fx.html#torch.fx.Tracer

If you want a real-world example of a custom tracer, check out FX's AST
Rewriter in `rewriter.py`. `RewritingTracer` inherits from Tracer but
overrides the `trace` function so that we can rewrite all calls to
`assert` to the more FX-friendly `torch.assert`.

Note that a call to `symbolic_trace(m)` is equivalent to
`GraphModule(m, Tracer().trace(m))`. (`Tracer` is the default
implementation of Tracer as defined in `symbolic_trace.py`.)
"""
# 如何创建和使用自定义追踪器。
# Tracer 实现 torch.fx.symbolic_trace 的符号追踪功能，可以通过继承它改变追踪过程。
# 本教程展示手写追踪器：只需重写少数方法，就能改变生成的计算图。完整说明见 Tracer 方法文档及原文链接。
# 实际案例可参考 rewriter.py 中的 AST 重写器：RewritingTracer 重写 trace，将 assert 调用转换为更适合 FX 的 torch.assert。
# symbolic_trace(m) 等价于 GraphModule(m, Tracer().trace(m))，其中 Tracer 为默认实现。



"""
Custom Tracer #1: Trace Through All `torch.nn.ReLU` Submodules

During symbolic tracing, some submodules are traced through and their
constituent ops are recorded; other submodules appear as an
atomic "call_module" Node in the IR. A module in this latter category
is called a "leaf module". By default, all modules in the PyTorch
standard library (`torch.nn`) are leaf modules. We can change this
by creating a custom Tracer and overriding `is_leaf_module`. In this
case, we'll keep the default behavior for all `torch.nn` Modules except
for `ReLU`.
"""
# 自定义追踪器 1：追踪所有 torch.nn.ReLU 子模块的内部实现。
# 符号追踪时，有些子模块会展开为组成操作，有些则作为原子的 call_module 节点保留；后者称为叶子模块。
# 默认情况下，torch.nn 标准库中的模块均为叶子模块。通过重写 is_leaf_module 可以改变此行为。
# 本例仅改变 ReLU 的行为，其余 torch.nn 模块维持默认设置。

class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

default_traced: GraphModule = symbolic_trace(M1())
"""
Tracing with the default tracer and calling `print_tabular` produces:

    opcode       name    target    args       kwargs
    -----------  ------  --------  ---------  --------
    placeholder  x       x         ()         {}
    call_module  relu_1  relu      (x,)       {}
    output       output  output    (relu_1,)  {}

"""
# 默认追踪器的表格输出：relu 表示为 call_module 节点。
# 各列依次为操作码、节点名、目标、位置参数和关键字参数。
default_traced.graph.print_tabular()

class LowerReluTracer(Tracer):
    def is_leaf_module(self, m : torch.nn.Module, qualname : str):
        if isinstance(m, torch.nn.ReLU):
            return False
        return super().is_leaf_module(m, qualname)

"""
Tracing with our custom tracer and calling `print_tabular` produces:

    opcode         name    target                             args       kwargs
    -------------  ------  ---------------------------------  ---------  ------------------
    placeholder    x       x                                  ()         {}
    call_function  relu_1  <function relu at 0x7f66f7170b80>  (x,)       {'inplace': False}
    output         output  output                             (relu_1,)  {}
"""
# 自定义追踪器的表格输出：ReLU 被展开为调用 relu 函数的 call_function 节点。
# 各列依次为操作码、节点名、目标、位置参数和关键字参数。
lower_relu_tracer = LowerReluTracer()
custom_traced_graph: Graph = lower_relu_tracer.trace(M1())
custom_traced_graph.print_tabular()



"""
Custom Tracer #2: Add an Extra Attribute to Each Node

Here, we'll override `create_node` so that we can add a new attribute to
each Node during its creation
"""
# 自定义追踪器 2：为每个节点增加属性。
# 重写 create_node，在创建各节点时添加新属性。

class M2(torch.nn.Module):
    def forward(self, a, b):
        return a + b

class TaggingTracer(Tracer):
    def create_node(self, kind : str, target : Union[str, Callable],
                    args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        n = super().create_node(kind, target, args, kwargs, name)
        n.tag = "foo"
        return n

custom_traced_graph: Graph = TaggingTracer().trace(M2())

def assert_all_nodes_have_tags(g: Graph) -> bool:
    for n in g.nodes:
        if not hasattr(n, "tag") or not n.tag == "foo":
            return False
    return True

# Prints "True"
# 打印 True。
print(assert_all_nodes_have_tags(custom_traced_graph))

