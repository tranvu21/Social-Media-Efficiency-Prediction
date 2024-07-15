import torch
import torch.nn as nn

class FollowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 16)
        self.model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, follow_count):
        follow_count = self.linear1(follow_count)
        x = torch.cat((x, follow_count), dim = 1)
        x = self.model(x)
        return x
