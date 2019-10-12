import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as opt

from helper import init_weights

class QNetwork(nn.Module):
    def __init__(self, n_s, n_a, n_h):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(n_s+n_a, n_h)
        self.linear2 = nn.Linear(n_h,n_h)
        self.linear3 = nn.Linear(n_h,1)

        self.apply(init_weights)

    def forward(self, s, a):
        x = torch.cat((s,a),1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x