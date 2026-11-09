# Time Sequence Prediction

时间序列预测

This is a toy example for beginners to start with. It helps learn both PyTorch and time sequence prediction. Two LSTMCell units are used in this example to learn some sine wave signals starting at different phases. After learning the sine waves, the network tries to predict the signal values in the future. The results are shown in the picture below.

这是面向初学者的小型示例，可以同时学习 PyTorch 和时间序列预测。本例使用两个 LSTMCell 单元，学习具有不同初始相位的正弦波。学习后，网络尝试预测未来的信号值，结果如下图所示。

## Usage

使用方法

```
python generate_sine_wave.py
python train.py
```

## Result

结果

The initial signal and the predicted results are shown in the image. We first give some initial signals (full line). The network will subsequently give some predicted results (dash line). It can be concluded that the network can generate new sine waves.

图中展示了初始信号和预测结果：先向网络提供一段初始信号（实线），再由网络预测后续信号（虚线）。可以看到，网络能够继续生成正弦波。

![image](https://cloud.githubusercontent.com/assets/1419566/24184438/e24f5280-0f08-11e7-8f8b-4d972b527a81.png)

初始信号与预测结果。
