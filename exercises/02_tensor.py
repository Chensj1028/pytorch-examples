import torch


class LinearModel:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def predict(self, x):
        # Select each feature column for all samples in the batch.
        # 取出所有样本的两列特征，一次计算整批样本的预测值。
        return self.w1 * x[:, 0] + self.w2 * x[:, 1] + self.b


def main():
    # Each row is one sample, and each column is one feature.
    # 每行是一个样本，每列是一个特征；输入形状为 [3, 2]。
    samples = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=torch.float32)

    model = LinearModel(2, -1, 1)
    predictions = model.predict(samples)

    print("输入形状:", samples.shape)
    print("预测结果:", predictions)
    print("输出形状:", predictions.shape)


if __name__ == "__main__":
    main()
