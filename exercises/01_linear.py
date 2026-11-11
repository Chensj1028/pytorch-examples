class LinearModel:
    def __init__(self, w1, w2, b):
        # TODO 1：把三个参数分别保存到对象属性中
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def predict(self, x):
        # TODO 2：使用 x[0]、x[1] 和对象属性计算并返回 y
        return self.w1 * x[0] + self.w2 * x[1] + self.b


def main():
    samples = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    # TODO 3：创建 LinearModel 对象，参数为 2、-1、1
    model = LinearModel(2, -1, 1)
    # TODO 4：遍历 samples，调用对象的 predict 方法并打印结果
    for sample in samples:
        y = model.predict(sample)
        print(f"Input: {sample}, Predicted Output: {y}")


if __name__ == "__main__":
    main()
