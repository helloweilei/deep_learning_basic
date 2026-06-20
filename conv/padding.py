import torch

def compute_conv(conv, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv(X)
    return Y.reshape(Y.shape[2:])

"""
卷积核大小为3*3,填充为1, 计算方式：填充大小 = (卷积核大小 - 1) / 2
可以通过stride设置步幅
"""
conv = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
X = torch.randn(8, 8)
Y = compute_conv(conv, X)
print(Y.shape)