import torch
from torch.fx import Proxy, Graph, GraphModule


'''
How to Create a Graph Using Proxy Objects Instead of Tracing

It's possible to directly create a Proxy object around a raw Node. This
can be used to create a Graph independently of symbolic tracing.

The following code demonstrates how to use Proxy with a raw Node to
append operations to a fresh Graph. We'll create two parameters (``x``
and ``y``), perform some operations on those parameters, then add
everything we created to the new Graph. We'll then wrap that Graph in
a GraphModule. Doing so creates a runnable instance of ``nn.Module``
where previously-created operations are represented in the Module's
``forward`` function.

By the end of the tutorial, we'll have added the following method to an
empty ``nn.Module`` class.

.. code-block:: python

    def forward(self, x, y):
        cat_1 = torch.cat([x, y]);  x = y = None
        tanh_1 = torch.tanh(cat_1);  cat_1 = None
        neg_1 = torch.neg(tanh_1);  tanh_1 = None
        return neg_1

'''
# 如何使用 Proxy 对象直接创建计算图，而不依赖符号追踪。
# 可以基于原始 Node 创建 Proxy，向新图追加操作。
# 本例创建输入 x、y，对它们执行操作，最后包装为 GraphModule，得到可运行的 nn.Module。
# 生成的 forward 依次执行拼接、tanh 和取负，再返回结果。


# Create a graph independently of symbolic tracing
# 独立于符号追踪，直接创建计算图。
graph = Graph()
tracer = torch.fx.proxy.GraphAppendingTracer(graph)

# Create raw Nodes
# 创建原始节点。
raw1 = graph.placeholder('x')
raw2 = graph.placeholder('y')

# Initialize Proxies using the raw Nodes and graph's default tracer
# 使用原始节点和计算图默认追踪器初始化 Proxy。
y = Proxy(raw1, tracer)
z = Proxy(raw2, tracer)
# y = Proxy(raw1)
# z = Proxy(raw2)

# Create other operations using the Proxies `y` and `z`
# 使用 Proxy 对象 y 和 z 创建其他操作。
a = torch.cat([y, z])
b = torch.tanh(a)
c = torch.neg(b)
# By using the graph's own appending tracer to create Proxies,
# 使用计算图自带的追加追踪器创建 Proxy 后，
# notice we can now use n-ary operators on operations without
# 可对操作使用多元运算符，
# multiple tracers being created at run-time (line 52) which leads
# 避免运行时创建多个追踪器而
# to errors # To try this out for yourself, replace lines 42, 43
# 导致错误。要自行尝试，可将原文所指的第 42、43 行
# with 44, 45
# 替换为第 44、45 行；添加注释后实际行号会变化。
z = torch.add(b, c)

# Create a new output Node and add it to the Graph. By doing this, the
# 创建新的输出节点并加入计算图。由于所有新节点
# Graph will contain all the Nodes we just created (since they're all
# 都与输出节点相连，计算图将包含
# linked to the output Node)
# 刚创建的所有节点。
graph.output(c.node)

# Wrap our created Graph in a GraphModule to get a final, runnable
# 使用 GraphModule 包装创建的计算图，得到最终可运行的
# `nn.Module` instance
# nn.Module 实例。
mod = GraphModule(torch.nn.Module(), graph)
