from torch import nn
import torch


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for module in self._modules.values():
            X = module(X)
        return X

def test():
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    X = torch.randn(size=(2, 20))
    Y = net(X)
    print(Y)

test()