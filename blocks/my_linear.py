from torch import nn
import torch

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.rand(out_features,))

    def forward(self, X):
        X = X @ self.weight.data + self.bias.data
        return X

class DropDimension(nn.Module):
    def __init__(self, dim):
        super(DropDimension, self).__init__()
        self.dim = dim

    def forward(self, X):
        return X.squeeze(self.dim)

class Sum2D(nn.Module):
    """将三维数据对前两个维度求和，降低到一维"""
    def __init__(self):
        super(Sum2D, self).__init__()

    def forward(self, X):
        # 对前两个维度（dim=0和dim=1）求和
        # 输入形状: (batch, feature, channels) -> 输出形状: (channels,)
        return X.sum(dim=0).sum(dim=0)

def test():
    # 测试 MyLinear
    net = MyLinear(20, 10)
    X = torch.randn(size=(2, 20))
    Y = net(X)
    print("MyLinear output shape:", Y.shape)
    print(Y)

    # 测试 Sum2D：三维数据对前两个维度求和
    sum_2d = Sum2D()
    X_3d = torch.randn(size=(2, 3, 4))  # 三维数据：(batch=2, feature=3, channels=4)
    Y_sum = sum_2d(X_3d)
    print("\nSum2D input shape:", X_3d.shape)
    print("Sum2D output shape:", Y_sum.shape)
    print("Sum2D output:", Y_sum)

if __name__ == '__main__':
    test()