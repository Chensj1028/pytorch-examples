"""
Recording Module Hierarchy With a Custom Tracer

In this example, we are going to define a custom `fx.Tracer` instance that--
for each recorded operation--also notes down the qualified name of the module
from which that operation originated. The _qualified name_ is the path to the
Module from the root module. More information about this concept can be
found in the documentation for `Module.get_submodule`:
https://github.com/pytorch/pytorch/blob/9f2aea7b88f69fc74ad90b1418663802f80c1863/torch/nn/modules/module.py#L385
"""
# 使用自定义追踪器记录模块层级。
# 本例定义自定义 fx.Tracer，为每个记录的操作同时保存来源模块的限定名称。
# 限定名称是从根模块到目标模块的路径，更多说明见原文中 Module.get_submodule 的文档链接。
import torch
import torch.fx
from typing import Any, Callable, Dict, Optional, Tuple

class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """
    # ModulePathTracer 是 FX 追踪器，会为每个操作额外记录其来源模块的限定名称。

    # The current qualified name of the Module being traced. The top-level
    # 当前被追踪模块的限定名称。顶层
    # module is signified by empty string. This is updated when entering
    # 模块用空字符串表示。进入 call_module 时更新，
    # call_module and restored when exiting call_module
    # 退出 call_module 时恢复。
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # 从 FX 节点到其来源模块限定名称的映射，
    # originated. This is recorded by `create_proxy` when recording an
    # 在 create_proxy 记录
    # operation
    # 操作时保存。
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).

        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        # 重写 Tracer.call_module：
        # 1）保存调用方的限定名称，以便稍后恢复；
        # 2）将当前模块限定名称写入 current_module_qualified_name，供 create_proxy 使用；
        # 3）调用常规 Tracer.call_module；
        # 4）恢复调用方的限定名称。
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        # 重写 Tracer.create_proxy：拦截每个操作的记录过程，
        # 将当前被追踪模块的限定名称保存到 node_to_originating_module。
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


# Testing: let's see how this works on a torchvision ResNet18 model
# 测试：在 torchvision 的 ResNet18 模型上查看效果。
import torchvision.models as models

# Model under test
# 待测模型。
rn18 = models.resnet18()

# Instantiate our ModulePathTracer and use that to trace our ResNet18
# 创建 ModulePathTracer，并用它追踪 ResNet18。
tracer = ModulePathTracer()
traced_rn18 = tracer.trace(rn18)

# Print (node, module qualified name) for every node in the Graph
# 打印计算图中每个节点及其所属模块的限定名称。
for node in traced_rn18.nodes:
    module_qualname = tracer.node_to_originating_module.get(node)
    print('Node', node, 'is from module', module_qualname)
"""
Node x is from module 
Node conv1 is from module conv1
Node bn1 is from module bn1
Node relu is from module relu
Node maxpool is from module maxpool
Node layer1_0_conv1 is from module layer1.0.conv1
Node layer1_0_bn1 is from module layer1.0.bn1
Node layer1_0_relu is from module layer1.0.relu
Node layer1_0_conv2 is from module layer1.0.conv2
Node layer1_0_bn2 is from module layer1.0.bn2
Node add is from module layer1.0
Node layer1_0_relu_1 is from module layer1.0.relu
Node layer1_1_conv1 is from module layer1.1.conv1
Node layer1_1_bn1 is from module layer1.1.bn1
Node layer1_1_relu is from module layer1.1.relu
Node layer1_1_conv2 is from module layer1.1.conv2
Node layer1_1_bn2 is from module layer1.1.bn2
Node add_1 is from module layer1.1
Node layer1_1_relu_1 is from module layer1.1.relu
Node layer2_0_conv1 is from module layer2.0.conv1
Node layer2_0_bn1 is from module layer2.0.bn1
Node layer2_0_relu is from module layer2.0.relu
Node layer2_0_conv2 is from module layer2.0.conv2
Node layer2_0_bn2 is from module layer2.0.bn2
Node layer2_0_downsample_0 is from module layer2.0.downsample.0
Node layer2_0_downsample_1 is from module layer2.0.downsample.1
Node add_2 is from module layer2.0
Node layer2_0_relu_1 is from module layer2.0.relu
Node layer2_1_conv1 is from module layer2.1.conv1
Node layer2_1_bn1 is from module layer2.1.bn1
Node layer2_1_relu is from module layer2.1.relu
Node layer2_1_conv2 is from module layer2.1.conv2
Node layer2_1_bn2 is from module layer2.1.bn2
Node add_3 is from module layer2.1
Node layer2_1_relu_1 is from module layer2.1.relu
Node layer3_0_conv1 is from module layer3.0.conv1
Node layer3_0_bn1 is from module layer3.0.bn1
Node layer3_0_relu is from module layer3.0.relu
Node layer3_0_conv2 is from module layer3.0.conv2
Node layer3_0_bn2 is from module layer3.0.bn2
Node layer3_0_downsample_0 is from module layer3.0.downsample.0
Node layer3_0_downsample_1 is from module layer3.0.downsample.1
Node add_4 is from module layer3.0
Node layer3_0_relu_1 is from module layer3.0.relu
Node layer3_1_conv1 is from module layer3.1.conv1
Node layer3_1_bn1 is from module layer3.1.bn1
Node layer3_1_relu is from module layer3.1.relu
Node layer3_1_conv2 is from module layer3.1.conv2
Node layer3_1_bn2 is from module layer3.1.bn2
Node add_5 is from module layer3.1
Node layer3_1_relu_1 is from module layer3.1.relu
Node layer4_0_conv1 is from module layer4.0.conv1
Node layer4_0_bn1 is from module layer4.0.bn1
Node layer4_0_relu is from module layer4.0.relu
Node layer4_0_conv2 is from module layer4.0.conv2
Node layer4_0_bn2 is from module layer4.0.bn2
Node layer4_0_downsample_0 is from module layer4.0.downsample.0
Node layer4_0_downsample_1 is from module layer4.0.downsample.1
Node add_6 is from module layer4.0
Node layer4_0_relu_1 is from module layer4.0.relu
Node layer4_1_conv1 is from module layer4.1.conv1
Node layer4_1_bn1 is from module layer4.1.bn1
Node layer4_1_relu is from module layer4.1.relu
Node layer4_1_conv2 is from module layer4.1.conv2
Node layer4_1_bn2 is from module layer4.1.bn2
Node add_7 is from module layer4.1
Node layer4_1_relu_1 is from module layer4.1.relu
Node avgpool is from module avgpool
Node flatten is from module 
Node fc is from module fc
Node output is from module None
"""
# 程序输出：每一行说明某个节点来自哪个模块。
# 例如 conv1 来自 conv1，layer1_0_conv1 来自 layer1.0.conv1；
# 根模块以空名称表示，最终输出节点的来源为 None。
