import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.apply(init_params)

    def forward(self, X):
        return self.out(self.relu(self.hidden(X)))

def init_params(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
