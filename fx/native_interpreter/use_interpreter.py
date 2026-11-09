import torch
import torch.fx
import operator

# Does this path not exist? Check that you've done the following:
# 如果路径不存在，请检查是否完成以下步骤：
# 1) Read README.md and follow the instructions to build libinterpreter.
# 1）阅读 README.md，按说明构建 libinterpreter。
# 2) If this file still does not exist after you've followed those instructions,
# 2）如果操作后仍无此文件，
#    check if it is under a different extension (e.g. `dylib` on mac or `dll` on
# 检查文件是否使用了其他扩展名，例如 macOS 的 dylib 或
#    windows).
# Windows 的 dll。
torch.classes.load_library('build/libinterpreter.so')

# This is what a lowering pass should look like: a function that takes
# 一个降低表示层次的转换阶段应是这样的函数：
# a valid nn.Module, symbolically traces it, lowers the Module to some
# 接收有效的 nn.Module，进行符号追踪，再转换为
# representation, and wraps that representation up into another
# 某种中间表示，将其封装到另一个
# nn.Module instance that handles dispatch to the compiled/lowered code.
# nn.Module 实例中，由后者调用编译或转换后的代码。
# This will ensure that this lowering transformation still fits into the
# 这样可以保证此转换仍符合
# PyTorch programming model and enables features like composing with other
# PyTorch 编程模型，并支持与其他转换组合、
# transformations and TorchScript compilation.
# 编译为 TorchScript 等功能。
def lower_to_elementwise_interpreter(orig_mod : torch.nn.Module) -> torch.nn.Module:
    # ===== Stage 1: Symbolic trace the module =====
    # 第 1 阶段：对模块进行符号追踪。
    mod = torch.fx.symbolic_trace(orig_mod)

    # ===== Stage 2: Lower GraphModule representation to the C++
    # 第 2 阶段：将 GraphModule 表示转换为 C++
    #       interpreter's instruction format ======
    # 解释器的指令格式。
    instructions = []
    constant_idx = 0
    constants = {}
    fn_input_names = []

    target_to_name = {
        operator.add : "add",
        operator.mul : "mul"
    }

    output_node : Optional[torch.fx.Node] = None
    # For each instruction, create a triple
    # 为每条指令创建一个三元组：
    # (instruction_name : str, inputs : List[str], output : str)
    # 指令名为字符串，输入为字符串列表，输出为字符串。
    # to feed into the C++ interpreter
    # 将其传给 C++ 解释器。
    for n in mod.graph.nodes:
        target, args, out_name = n.target, n.args, n.name
        assert len(n.kwargs) == 0, "kwargs currently not supported"

        if n.op == 'placeholder':
            # Placeholders specify function argument names. Save these
            # 占位符指定函数参数名称；保存这些名称，
            # for later when we generate the wrapper GraphModule
            # 供之后生成包装用 GraphModule 时使用。
            fn_input_names.append(target)
        elif n.op == 'call_function':
            assert target in target_to_name, "Unsupported call target " + target
            arg_names = []
            for arg in args:
                if not isinstance(arg, torch.fx.Node):
                    # Pull out constants. These constants will later be
                    # 提取常量，稍后通过 add_constant()
                    # fed to the interpreter C++ object via add_constant()
                    # 将这些常量传给 C++ 解释器对象。
                    arg_name = f'constant_{constant_idx}'
                    constants[arg_name] = torch.Tensor(
                        [arg] if isinstance(arg, numbers.Number) else arg)
                    arg_names.append(arg_name)
                    constant_idx += 1
                else:
                    arg_names.append(arg.name)
            instructions.append((target_to_name[target], arg_names, out_name))
        elif n.op == 'output':
            if output_node is not None:
                raise RuntimeError('Multiple output nodes!')
            output_node = n
        else:
            raise RuntimeError('Unsupported opcode ' + n.op)

    interpreter = torch.classes.NativeInterpretation.ElementwiseInterpreter()
    # Load constants
    # 加载常量。
    for k, v in constants.items():
        interpreter.add_constant(k, v)
    # Specify names for positional input arguments
    # 指定位置输入参数的名称。
    interpreter.set_input_names(fn_input_names)
    # Load instructions
    # 加载指令。
    interpreter.set_instructions(instructions)
    # Specify name for single output
    # 指定唯一输出的名称。
    assert isinstance(output_node.args[0], torch.fx.Node)
    interpreter.set_output_name(output_node.args[0].name)

    # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====
    # 第 3 阶段：创建包装解释器的 GraphModule。
    class WrapperModule(torch.nn.Module):
        def __init__(self, interpreter):
            super().__init__()
            self.interpreter = interpreter

    wrapper = WrapperModule(interpreter)

    # Create a forward() function that is compatible with TorchScript compilation.
    # 创建可兼容 TorchScript 编译的 forward() 函数。
    # Create a graph that: 1) Takes function arguments 2) Invokes the interpreter
    # 创建一张计算图：1）接收函数参数；2）调用解释器；
    # 3) Returns the specified return value
    # 3）返回指定的结果。

    graph = torch.fx.Graph()
    # Add placeholders for fn inputs
    # 为函数输入添加占位符。
    placeholder_nodes = []
    for name in fn_input_names:
        placeholder_nodes.append(graph.create_node('placeholder', name))

    # Get the interpreter object
    # 获取解释器对象。
    interpreter_node = graph.create_node('get_attr', 'interpreter')

    # Add a node to call the interpreter instance
    # 添加调用解释器实例的节点。
    output_node = graph.create_node(
        op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))

    # Register output
    # 注册输出。
    graph.output(output_node)

    graph.lint(wrapper)

    # Return final GraphModule!!!
    # 返回最终的 GraphModule。
    return torch.fx.GraphModule(wrapper, graph)

class MyElementwiseModule(torch.nn.Module):
    def forward(self, x, y):
        return x * y + y

mem = MyElementwiseModule()
lowered = lower_to_elementwise_interpreter(mem)
print(lowered.code)
# The lowered module can also be compiled into TorchScript
# 转换后的模块也可以编译为 TorchScript。
scripted = torch.jit.script(lowered)
print(scripted.graph)

# Stress test correctness
# 进行正确性压力测试。
for _ in range(50):
    x, y = torch.randn(10, 20, 30), torch.randn(10, 20, 30)
    torch.testing.assert_allclose(lowered(x, y), mem(x, y))
    torch.testing.assert_allclose(scripted(x, y), mem(x, y))
