import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import hyp

class PolicyNetwork(nn.Module):
    def __init__(self,s_dim,a_dim,h_dim):
        super(PolicyNetwork,self).__init__()

        self.linear1 = nn.Linear(s_dim,h_dim)
        self.linear2 = nn.Linear(h_dim,h_dim)
        self.linear3a = nn.Linear(h_dim,a_dim)
        self.linear3b = nn.Linear(h_dim,a_dim)

        # self.apply(init_weights)

    def forward(self,s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        mean = self.linear3a(x)
        log_std = self.linear3b(x)

        log_std = torch.clamp(log_std, min=hyp.MIN_LOG, max=hyp.MAX_LOG)
        
        return mean, log_std

    def sample_action(self,s):
        mean, log_std = self.forward(s)
        std = log_std.exp()

        normal = Normal(0,1)
        xi = normal.sample()
        u = mean + std*xi.to(hyp.device)
        a = torch.tanh(u)

        log_pi = Normal(mean,std).log_prob(u) - torch.log(1 - a.pow(2) + hyp.EPSILON)

        return a, log_pi