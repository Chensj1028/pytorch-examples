# 时间序列预测

[English original](README.md)

> 中文说明；代码块中的命令、代码及程序输出保留原样。

这是面向初学者的小型示例，可以同时学习 PyTorch 和时间序列预测。本例使用两个 LSTMCell 单元，学习具有不同初始相位的正弦波。学习后，网络尝试预测未来的信号值，结果如下图所示。

## 使用方法

```
python generate_sine_wave.py
python train.py
```

## 结果

图中展示了初始信号和预测结果：先向网络提供一段初始信号（实线），再由网络预测后续信号（虚线）。可以看到，网络能够继续生成正弦波。

![初始信号与预测结果](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)
