import torch
import torch.fx as fx

# An inverse mapping is one that takes a function f(x) and returns a function g
# 逆映射接收函数 f(x)，返回函数 g，
# such that f(g(x)) == x. For example,since log(exp(x)) == x, exp and log are
# 使 f(g(x)) == x。例如 log(exp(x)) == x，因此 exp 和 log
# inverses.
# 互为逆函数。

invert_mapping = {}
def add_inverse(a, b):
    invert_mapping[a] = b
    invert_mapping[b] = a
inverses = [
    (torch.sin, torch.arcsin),
    (torch.cos, torch.arccos),
    (torch.tan, torch.arctan),
    (torch.exp, torch.log),
]
for a, b in inverses:
    add_inverse(a, b)

# The general strategy is that we walk the graph backwards, transforming each
# 整体策略是逆向遍历计算图，将每个
# node into its inverse. To do so, we swap the outputs and inputs of the
# 节点变为其逆操作：交换函数的输入与输出，
# functions, and then we look up its inverse in `invert_mapping`. Note that
# 再从 invert_mapping 中查找逆函数。注意，
# this transform assumes that all operations take in only one input and return
# 此变换假设所有操作都只有一个输入，
# one output.
# 并返回一个输出。
def invert(model: torch.nn.Module) -> torch.nn.Module:
    fx_model = fx.symbolic_trace(model)
    new_graph = fx.Graph()  # As we're building up a new graph
    # 这里正在构建一张新图。
    env = {}
    for node in reversed(fx_model.graph.nodes):
        if node.op == 'call_function':
            # This creates a node in the new graph with the inverse function,
            # 在新图中创建调用逆函数的节点，
            # and passes `env[node.name]` (i.e. the previous output node) as
            # 并将 env[node.name]，即之前的输出节点，作为
            # input.
            # 输入。
            new_node = new_graph.call_function(invert_mapping[node.target], (env[node.name],))
            env[node.args[0].name] = new_node
        elif node.op == 'output':
            # We turn the output into an input placeholder
            # 将输出转换为输入占位符。
            new_node = new_graph.placeholder(node.name)
            env[node.args[0].name] = new_node
        elif node.op == 'placeholder':
            # We turn the input placeholder into an output
            # 将输入占位符转换为输出。
            new_graph.output(env[node.name])
        else:
            raise RuntimeError("Not implemented")

    new_graph.lint()
    return fx.GraphModule(fx_model, new_graph)


def f(x):
    return torch.exp(torch.tan(x))

res = invert(f)
print(res.code)
"""
def forward(self, output):
    log_1 = torch.log(output);  output = None
    arctan_1 = torch.arctan(log_1);  log_1 = None
    return arctan_1
"""
# 生成的 forward 示例：先计算对数，再计算反正切，返回结果。
print(f(res((torch.arange(5) + 1))))  # [1., 2., 3., 4, 5.]
