#include <torch/script.h>

#include <sstream>

// ElementwiseInterpreter is a class that takes in a list of Instructions
// ElementwiseInterpreter 类接收指令列表，
// (represented as triples (op, inputs, outputs)) and executes them in a
// 每条指令由 (op, inputs, outputs) 三元组表示，
// C++-based interpreter loop. For brevity, this interpreter only supports
// 并在基于 C++ 的解释器循环中执行。为保持简洁，解释器只支持
// two operations: element-wise add and element-wise mul.
// 两种操作：逐元素加法和逐元素乘法。
struct ElementwiseInterpreter : torch::CustomClassHolder {
  using InstructionType =
      std::tuple<std::string /*op*/, std::vector<std::string> /*inputs*/,
      // 操作。
      // 输入。
                 std::string /*output*/>;
                 // 输出。

  ElementwiseInterpreter() {}

  // Load a list of instructions into the interpreter. As specified above,
  // 将指令列表加载到解释器。按前面的约定，
  // instructions specify the operation (currently support "add" and "mul"),
  // 指令需要指定操作（目前支持 add 和 mul）、
  // the names of the input values, and the name of the single output value
  // 输入值的名称，以及本条指令
  // from this instruction
  // 唯一输出值的名称。
  void setInstructions(std::vector<InstructionType> instructions) {
    instructions_ = std::move(instructions);
  }

  // Add a constant. The interpreter maintains a set of constants across
  // 添加常量。解释器在多次调用之间保存一组常量，
  // calls. They are keyed by name, and constants can be referenced in
  // 以名称作为键，指令可以通过
  // Instructions by the name specified
  // 指定名称引用常量。
  void addConstant(const std::string &name, at::Tensor value) {
    constants_.insert_or_assign(name, std::move(value));
  }

  // Set the string names for the positional inputs to the function this
  // 设置此解释器所代表函数的位置输入参数名称。
  // interpreter represents. When invoked, the interpreter will assign
  // 调用时，解释器会将
  // the positional inputs to the names in the corresponding position in
  // 各位置输入绑定到 input_names 中
  // input_names.
  // 对应位置的名称。
  void setInputNames(std::vector<std::string> input_names) {
    input_names_ = std::move(input_names);
  }

  // Specify the output name for the function this interpreter represents. This
  // 指定此解释器所代表函数的输出名称。
  // should match the "output" field of one of the instructions in the
  // 名称应与指令列表中某条指令的 output 字段
  // instruction list, typically the last instruction.
  // 匹配，通常是最后一条指令。
  void setOutputName(std::string output_name) {
    output_name_ = std::move(output_name);
  }

  // Invoke this interpreter. This takes a list of positional inputs and returns
  // 调用解释器：接收位置输入列表，返回
  // a single output. Currently, inputs and outputs must all be Tensors.
  // 单个输出。目前输入和输出都必须为张量。
  at::Tensor __call__(std::vector<at::Tensor> inputs) {
    // Environment to hold local variables
    // 用于保存局部变量的环境。
    std::unordered_map<std::string, at::Tensor> environment;

    // Load inputs according to the specified names
    // 按照指定名称加载输入。
    if (inputs.size() != input_names_.size()) {
      std::stringstream err;
      err << "Expected " << input_names_.size() << " inputs, but got "
          << inputs.size() << "!";
      throw std::runtime_error(err.str());
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      environment[input_names_[i]] = inputs[i];
    }

    if (!output_name_) {
      throw std::runtime_error("Output name not specified!");
    }

    for (InstructionType &instr : instructions_) {
      // Retrieve all input values for this op
      // 获取当前操作的所有输入值。
      std::vector<at::Tensor> inputs;
      for (const auto &input_name : std::get<1>(instr)) {
        // Operator output values shadow constants.
        // 操作输出值会遮蔽同名常量。
        // Imagine all constants are defined in statements at the beginning
        // 可以将所有常量视为定义在函数开头的语句，
        // of a function (a la K&R C). Any definition of an output value must
        // 类似 K&R C；任何输出值的定义
        // necessarily come after constant definition in textual order. Thus,
        // 在文本顺序上都位于常量定义之后。因此，
        // We look up values in the environment first then the constant table
        // 查找时先访问局部环境，再访问常量表，
        // second to implement this shadowing behavior
        // 以实现这种遮蔽行为。
        if (environment.find(input_name) != environment.end()) {
          inputs.push_back(environment.at(input_name));
        } else if (constants_.find(input_name) != constants_.end()) {
          inputs.push_back(constants_.at(input_name));
        } else {
          std::stringstream err;
          err << "Instruction referenced unknown value " << input_name << "!";
          throw std::runtime_error(err.str());
        }
      }

      // Run the specified operation
      // 执行指定操作。
      at::Tensor result;
      const auto &op = std::get<0>(instr);
      if (op == "add") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for add op!");
        }
        result = inputs[0] + inputs[1];
      } else if (op == "mul") {
        if (inputs.size() != 2) {
          throw std::runtime_error("Unexpected number of inputs for mul op!");
        }
        result = inputs[0] * inputs[1];
      } else {
        std::stringstream err;
        err << "Unknown operator " << op << "!";
        throw std::runtime_error(err.str());
      }

      // Write back result into environment
      // 将结果写回环境。
      const auto &output_name = std::get<2>(instr);
      environment[output_name] = std::move(result);
    }

    if (!environment.count(*output_name_)) {
      std::stringstream err;
      err << "Execution expected an output value with name ";
      err << *output_name_;
      err << " but no instruction produced a value with that name!";
      throw std::runtime_error(err.str());
    }

    return environment.at(*output_name_);
  }

  // Ser/De infrastructure. See
  // 序列化与反序列化基础设施；参见相关链接，
  // https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html#defining-serialization-deserialization-methods-for-custom-c-classes
  // for more info.
  // 了解更多信息。

  // This is the type we will use to marshall information on disk during
  // 此类型用于序列化和反序列化时
  // ser/de. It is a simple tuple composed of primitive types and simple
  // 传递磁盘数据，是由基本类型及 vector、optional、dict
  // collection types like vector, optional, and dict.
  // 等简单集合类型组成的元组。
  using SerializationType =
      std::tuple<std::vector<std::string> /*input_names_*/,
                 c10::optional<std::string> /*output_name_*/,
                 c10::Dict<std::string, at::Tensor> /*constants_*/,
                 std::vector<InstructionType> /*instructions_*/
                 >;

  // This function yields the SerializationType instance for `this`.
  // 此函数返回当前对象的 SerializationType 实例。
  SerializationType __getstate__() const {
    return SerializationType{input_names_, output_name_, constants_,
                             instructions_};
  }

  // This function will create an instance of `ElementwiseInterpreter` given
  // 此函数根据 SerializationType 实例创建
  // an instance of `SerializationType`.
  // ElementwiseInterpreter 实例。
  static c10::intrusive_ptr<ElementwiseInterpreter>
  __setstate__(SerializationType state) {
    auto instance = c10::make_intrusive<ElementwiseInterpreter>();
    std::tie(instance->input_names_, instance->output_name_,
             instance->constants_, instance->instructions_) = std::move(state);
    return instance;
  }

  // Class members
  // 类成员。
  std::vector<std::string> input_names_;
  c10::optional<std::string> output_name_;
  c10::Dict<std::string, at::Tensor> constants_;
  std::vector<InstructionType> instructions_;
};

// Register ElementwiseInterpreter as callable from Python
// 注册 ElementwiseInterpreter，使 Python
// and TorchScript.
// 和 TorchScript 可以调用。
TORCH_LIBRARY(NativeInterpretation, m) {
  m.class_<ElementwiseInterpreter>("ElementwiseInterpreter")
      .def(torch::init<>())
      .def("set_instructions", &ElementwiseInterpreter::setInstructions)
      .def("add_constant", &ElementwiseInterpreter::addConstant)
      .def("set_input_names", &ElementwiseInterpreter::setInputNames)
      .def("set_output_name", &ElementwiseInterpreter::setOutputName)
      .def("__call__", &ElementwiseInterpreter::__call__)
      .def_pickle(
          /* __getstate__ */
          [](const c10::intrusive_ptr<ElementwiseInterpreter> &self) {
            return self->__getstate__();
          },
          /* __setstate__ */
          [](ElementwiseInterpreter::SerializationType state) {
            return ElementwiseInterpreter::__setstate__(std::move(state));
          });
}
