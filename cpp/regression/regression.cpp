#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define POLY_DEGREE 4

// Builds features i.e. a matrix with columns [x, x^2, x^3, x^4].
// 构建特征矩阵，各列为 [x, x^2, x^3, x^4]。
torch::Tensor make_features(torch::Tensor x) {
  x = x.unsqueeze(1);
  std::vector<torch::Tensor> xs;
  for (int64_t i = 0; i < POLY_DEGREE; ++i)
    xs.push_back(x.pow(i + 1));
  return torch::cat(xs, 1);
}

// Approximated function.
// 拟合得到的函数。
torch::Tensor f(
    torch::Tensor x,
    torch::Tensor W_target,
    torch::Tensor b_target) {
  return x.mm(W_target) + b_target.item();
}

// Creates a string description of a polynomial.
// 创建描述多项式的字符串。
std::string poly_desc(torch::Tensor W, torch::Tensor b) {
  auto size = W.size(0);
  std::ostringstream stream;

  stream << "y = ";
  for (int64_t i = 0; i < size; ++i)
    stream << W[i].item<float>() << " x^" << size - i << " ";
  stream << "+ " << b[0].item<float>();
  return stream.str();
}

// Builds a batch i.e. (x, f(x)) pair.
// 构建一个批次，即一组 (x, f(x)) 数据对。
std::pair<torch::Tensor, torch::Tensor> get_batch(
    torch::Tensor W_target,
    torch::Tensor b_target,
    int64_t batch_size = 32) {
  auto random = torch::randn({batch_size});
  auto x = make_features(random);
  auto y = f(x, W_target, b_target);
  return std::make_pair(x, y);
}

int main() {
  auto W_target = torch::randn({POLY_DEGREE, 1}) * 5;
  auto b_target = torch::randn({1}) * 5;

  // Define the model and optimizer
  // 定义模型和优化器。
  auto fc = torch::nn::Linear(W_target.size(0), 1);
  torch::optim::SGD optim(fc->parameters(), .1);

  float loss = 0;
  int64_t batch_idx = 0;

  while (++batch_idx) {
    // Get data
    // 获取数据。
    torch::Tensor batch_x, batch_y;
    std::tie(batch_x, batch_y) = get_batch(W_target, b_target);

    // Reset gradients
    // 清空梯度。
    optim.zero_grad();

    // Forward pass
    // 前向传播。
    auto output = torch::smooth_l1_loss(fc(batch_x), batch_y);
    loss = output.item<float>();

    // Backward pass
    // 反向传播。
    output.backward();

    // Apply gradients
    // 应用梯度更新。
    optim.step();

    // Stop criterion
    // 停止条件。
    if (loss < 1e-3f)
      break;
  }

  std::cout << "Loss: " << loss << " after " << batch_idx << " batches"
            << std::endl;
  std::cout << "==> Learned function:\t"
            << poly_desc(fc->weight.view({-1}), fc->bias) << std::endl;
  std::cout << "==> Actual function:\t"
            << poly_desc(W_target.view({-1}), b_target) << std::endl;

  return 0;
}
