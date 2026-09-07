import torch
from torch import nn

def corr2d(X, K):
    """计算二维卷积"""
    w, h = K.shape
    Y = torch.zeros((X.shape[0] - w + 1, X.shape[1] - h + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+w, j:j+h] * K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

def train_conv2d():
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2))
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    lr = 3e-2
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    for i in range(16):
        Y_hat = conv2d(X)
        l = ((Y_hat - Y) ** 2).sum()
        conv2d.zero_grad()
        l.backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {l.item():.3f}')
    print(f'weight: {conv2d.weight.data.squeeze()}', f'Actual K: {K.tolist()}', sep='\n')

# K: (cin, kh, kw)
def corr2d_multi_in(X, K):
    return sum(
        [corr2d(x, k) for x, k in zip(X, K)]
    )

# 多通道输出
# K: (cout, cin, kh, kw)
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def test_corr2d_multi_channel():
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    print(corr2d_multi_in(X, K))
    K2 = torch.stack([K, K + 1, K + 2], 0)
    print('多通道输出： ', corr2d_multi_in_out(X, K2), sep='\n')

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

def le_net():
    net = nn.Sequential(
        # in_channels, out_channels, kernel_size
        nn.Conv2d(1, 6, kernel_size=(5, 5)),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(6, 16, kernel_size=(5, 5)),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    return net

def test():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))

    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)

if __name__ == '__main__':
    # train_conv2d()
    test_corr2d_multi_channel()