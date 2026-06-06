import torch

from blocks.mlp import MLP
def main():
    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    print(Y)

if __name__ == "__main__":
    main()
