import torch
import torch.fx
"""
In this example we are going do define a library of
"composite" operations. Composite operations are those
that are defined as callable functions that are composed
of several other operations in their implementation.

Composite operations allow you to choose at what level
of abstraction you want to interpret/manipulate the
code. We show that we can provide a function to inline
these functions as well as use a custom Tracer to auto-
matically inline such functions.

Composite operations can be useful for exposing higher-
level context to a backend/transform while still
maintaining the ability to examine things at a more
fine-grained level.
"""
# 本例定义复合操作库：每个复合操作是由多个底层操作组成的可调用函数。
# 复合操作允许选择解释或修改代码时的抽象层次。
# 既可以通过函数显式内联，也可以让自定义 Tracer 自动内联。
# 这样既能向后端或转换阶段提供高层上下文，也能保留检查细粒度操作的能力。


def sigmoid_lowp(x : torch.Tensor):
    x = x.float()
    x = x.sigmoid()
    return x.half()

# wrap() indicates that the passed-in function should always
# wrap() 表示传入的函数应始终
# be recorded as a call_function node rather than being traced
# 记录为 call_function 节点，而不追踪其
# through. Later, we will see how we can:
# 内部实现。后面将展示如何：
# a. Inline the implementation of such a function and
# a．内联展开该函数的实现；
# b. Define a tracer that automatically traces through such a function
# b．定义自动追踪该函数内部实现的追踪器。
torch.fx.wrap(sigmoid_lowp)

def add_lowp(a : torch.Tensor, b : torch.Tensor):
    a, b = a.float(), b.float()
    c = a + b
    return c.half()

torch.fx.wrap(add_lowp)


# Let's see what happens when we symbolically trace through some code
# 下面看看符号追踪包含这些函数的
# that uses these functions
# 代码时会发生什么。

class Foo(torch.nn.Module):
    def forward(self, x, y):
        x = sigmoid_lowp(x)
        y = sigmoid_lowp(y)
        return add_lowp(x, y)


traced = torch.fx.symbolic_trace(Foo())
print(traced.code)
"""
def forward(self, x, y):
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
    add_lowp = __main___add_lowp(sigmoid_lowp, sigmoid_lowp_1);  sigmoid_lowp = sigmoid_lowp_1 = None
    return add_lowp
"""
# 生成的代码保留了两次 sigmoid_lowp 调用和一次 add_lowp 调用。

# Notice that the calls to `sigmoid_lowp` and `add_lowp`
# 注意，sigmoid_lowp 和 add_lowp 的调用
# appear literally in the trace; they are not traced through
# 按原样出现在追踪结果中，其内部实现没有被追踪。


# ***** Inlining calls *****
# 内联函数调用。
# Now let's define a function that allows for inlining these calls
# 现在定义一个函数，使这些调用能在
# during graph manipulation.
# 修改计算图时内联展开。

def inline_lowp_func(n : torch.fx.Node):
    # If we find a call to a function in our "lowp" module, inline it
    # 如果发现调用 lowp 模块中的函数，就将其内联。
    if n.op == 'call_function' and n.target.__module__ == inline_lowp_func.__module__:
        # We want to insert the operations comprising the implementation of the
        # 将组成该函数实现的操作插入到
        # function before the function itself. Then, we can swap the output value
        # 函数调用节点之前，然后将函数调用的输出值
        # of the function call with the output value for its implementation nodes
        # 替换为实现节点的输出值。
        tracer = torch.fx.proxy.GraphAppendingTracer(n.graph)
        with n.graph.inserting_before(n):
            # We can inline code by using `fx.Proxy` instances.
            # 可通过 fx.Proxy 实例内联代码。
            # map_arg traverses all aggregate types and applies the given function
            # map_arg 遍历各种聚合数据类型，并对
            # to Node instances in the data structure. In this case, we are applying
            # 结构中的 Node 实例应用指定函数；这里应用的是
            # the fx.Proxy constructor.
            # fx.Proxy 构造函数。
            proxy_args = torch.fx.node.map_arg(n.args, lambda x: torch.fx.Proxy(x, tracer))
            proxy_kwargs = torch.fx.node.map_arg(n.kwargs, lambda x: torch.fx.Proxy(x, tracer))
            # Call the function itself with proxy arguments. This will emit
            # 使用 Proxy 参数调用函数本身，生成
            # nodes in the graph corresponding to the operations in the im-
            # 与函数实现中各个操作对应的
            # plementation of the function
            # 计算图节点。
            output_proxy = n.target(*proxy_args, **proxy_kwargs)
            # Now replace the original node's uses with the output node of
            # 用函数实现的输出节点替换
            # the implementation.
            # 原节点的所有使用位置。
            node.replace_all_uses_with(output_proxy.node)
            # Delete the old node
            # 删除旧节点。
            node.graph.erase_node(node)

for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target is sigmoid_lowp:
        inline_lowp_func(node)

# Don't forget to recompile after graph manipulation
# 修改计算图后，别忘了重新编译。
traced.recompile()

print(traced.code)
"""
def forward(self, x, y):
    float_1 = x.float();  x = None
    sigmoid = float_1.sigmoid();  float_1 = None
    half = sigmoid.half();  sigmoid = None
    float_2 = y.float();  y = None
    sigmoid_1 = float_2.sigmoid();  float_2 = None
    half_1 = sigmoid_1.half();  sigmoid_1 = None
    add_lowp = __main___add_lowp(half, half_1);  half = half_1 = None
    return add_lowp
"""
# 生成的代码内联了 sigmoid_lowp，将其展开为 float、sigmoid、half，仍保留 add_lowp 调用。

# At this point, the implementation of `sigmoid_lowp` has been substituted
# 至此，所有对 sigmoid_lowp 的调用
# in for all of the calls to that function.
# 都已被该函数的实现替换。

# ***** Inlining calls during tracing *****
# 在追踪过程中内联调用。
# Now we are going to define a custom tracer that can selectively inline
# 接下来定义一个自定义追踪器，能够在追踪时
# calls to certain composite operations on-the-fly.
# 选择性内联某些复合操作。

# New instance of our module
# 创建模块的新实例。
f = Foo()

class InliningTracer(torch.fx.Tracer):
    FNS_TO_INLINE = [add_lowp]

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if kind == 'call_function' and target in self.FNS_TO_INLINE:
            tracer = torch.fx.proxy.GraphAppendingTracer(self.graph)
            # Trace through the implementation of the function rather than
            # 追踪函数的内部实现，
            # create a node
            # 而不是创建一个调用节点。
            proxy_args = torch.fx.node.map_arg(args, lambda x: torch.fx.Proxy(x, tracer))
            proxy_kwargs = torch.fx.node.map_arg(kwargs, lambda x: torch.fx.Proxy(x, tracer))
            return target(*proxy_args, **proxy_kwargs).node
        else:
            return super().create_node(kind, target, args, kwargs, name, type_expr)


tracer = InliningTracer()
graph = tracer.trace(f)
module = torch.fx.GraphModule(f, graph)
print(module.code)
"""
def forward(self, x, y):
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
    float_1 = sigmoid_lowp.float();  sigmoid_lowp = None
    float_2 = sigmoid_lowp_1.float();  sigmoid_lowp_1 = None
    add = float_1 + float_2;  float_1 = float_2 = None
    half = add.half();  add = None
    return half
"""
# 生成的代码保留 sigmoid_lowp 调用，并将 add_lowp 展开为 float 转换、加法和 half 转换。

# As you can see, the implementation for `add_lowp` has been
# 可以看到，使用 InliningTracer 追踪时，
# inlined in the course of tracing with our InliningTracer.
# add_lowp 的实现已被内联展开。
# Such functionality can be used to, for example, implement
# 这种功能可以用于实现某种后端：
# a backend that wants to see the lowered form of some operations
# 对于一部分操作使用降低层次后的表示，
# but the high-level form of another.
# 对于另一部分操作仍保留高层表示。

# ***** Future direction *****
# 后续方向。
#
# We may define an API, such as `Tracer.is_leaf_function`, that
# 未来可以定义 Tracer.is_leaf_function 之类的 API，
# Tracer implementers can use to more easily specify the inlining
# 让追踪器实现者更方便地指定
# behavior implemented in InliningTracer. Such a method would return
# InliningTracer 中的内联行为。该方法默认返回
# True by default, but a Tracer can override it and return `False` for
# True；对于希望继续追踪其内部实现的函数，
# functions the Tracer wants to be traced through.
# 追踪器可重写方法并返回 False。
