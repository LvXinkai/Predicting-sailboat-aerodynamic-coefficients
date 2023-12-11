import torch.nn as nn


# Operational Parameters and Aerodynamic Coefficients Mapping Block
class AlphaToBeta(nn.Module):
    def __init__(self):
        super(AlphaToBeta, self).__init__()
        self.linear1 = nn.Linear(3, 128)
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(512, 1024)
        self.linear4 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))

        return x


class BetaToR(nn.Module):
    def __init__(self):
        super(BetaToR, self).__init__()
        self.linear4 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear1 = nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear2(x))
        x = self.linear1(x)

        return x
