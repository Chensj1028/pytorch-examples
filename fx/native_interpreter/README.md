# Converting PyTorch Code to a Native Runtime With FX and TorchScript Custom Classes

使用 FX 和 TorchScript 自定义类，将 PyTorch 代码转换到原生运行时

In this example, we are going to build a pipeline that does the following things:

本示例构建如下处理流程：

1. Converts (or “lowers”) code in a PyTorch module into another representation (we will define the representation within the example)

   将 PyTorch 模块中的代码转换（也称为降低，lowering）成另一种表示形式，该形式将在示例中定义。

2. Registers an interpreter for that code representation that can be used in TorchScript or Python

   为这种表示形式注册解释器，使其可以在 TorchScript 或 Python 中使用。

3. Wrap the converted code into a format that can still be used in TorchScript compilation.

   将转换后的代码封装成仍能参与 TorchScript 编译的形式。

We are going to build up a trivial interpreter for this example, but you can imagine extending the same process to work with more sophisticated backends, ones which may do code optimization or offloading to an accelerator.

示例只实现一个简单解释器，但可以沿用相同流程扩展到更复杂的后端，例如支持代码优化或将计算交给加速设备的后端。

We will be using [TorchScript custom classes](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html) to expose this Interpreter to Python and TorchScript. You may want to review that tutorial and documentation before reading this example project.

本例通过 [TorchScript 自定义类](https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html)，将解释器开放给 Python 和 TorchScript 使用。阅读本项目前，建议先了解该教程和文档。

### Defining the Interpreter

定义解释器

We define the interpreter in `interpreter.cpp`. This interpreter is very limited: it only supports two element-wise operations (`add` and `mul`) and it only supports `Tensor` values. When this interpreter runs code, it iterates through the list of instructions and simply calls the appropriate PyTorch operator from C++.

解释器定义在 `interpreter.cpp` 中。它只支持逐元素 `add`、`mul` 两种运算，以及 `Tensor` 类型的值。执行代码时，解释器依次遍历指令列表，在 C++ 中调用对应的 PyTorch 算子。

To build the interpreter into a shared-object file to be loaded in for use, use the following commands from this example’s root:

在本示例根目录执行以下命令，将解释器构建为可加载的共享库：

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
$ make -j
```

After the build finishes, you should see `build/libinterpreter.so` (or with a different extension depending on your OS). We will use this dynamic library next when we load it up into a process to be used in execution.

构建完成后，应生成 `build/libinterpreter.so`；不同操作系统的扩展名可能不同。后续将此动态库加载到进程中执行。

### Defining the Transformation

定义转换过程

We define the code that transforms a `PyTorch` module to the format the interpreter understands in `use_interpreter.py`. Note that that file loads in the shared object we built in the previous step via a `torch.classes.load_library` call. `use_interpreter.py` contains driver code and the end that can be directly run to test the lowering transformation.

`use_interpreter.py` 定义了将 PyTorch 模块转换为解释器所理解格式的代码。文件通过 `torch.classes.load_library` 加载前一步生成的共享库。文件末尾提供了运行代码，可以直接执行以测试 lowering 转换。

### Questions, Comments, Feedback

问题、讨论与反馈

Please direct questions and discussion to the [PyTorch forums](https://discuss.pytorch.org/). To report any issues with PyTorch (including FX and custom classes), please use the [issue tracker](https://github.com/pytorch/pytorch/issues).

问题和讨论请前往 [PyTorch 论坛](https://discuss.pytorch.org/)。如需报告 PyTorch 问题，包括 FX 和自定义类的问题，请使用 [issue 跟踪页面](https://github.com/pytorch/pytorch/issues)。
