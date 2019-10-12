import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as opt

from helper import init_weights

# Hyperparameters
LOG_MIN = -20
LOG_MAX = 2
EPSILON = 1e-6

class PolicyNetwork(nn.Module):
    def __init__(self, n_s, n_h, n_a, action_space):
        super(PolicyNetwork,self).__init__()

        self.linear1 = nn.Linear(n_s,n_h)
        self.linear2 = nn.Linear(n_h,n_h)
        self.mean_linear = nn.Linear(n_h,n_a)
        self.log_std_linear = nn.Linear(n_h,n_a)

        self.apply(init_weights)

        # normalize actions
        self.action_scale = (action_space.high-action_space.low) / 2
        self.action_bias = (action_space.high+action_space.low) / 2

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # restrict log value to range
        log_std = log_std.clamp(min=LOG_MIN, max=LOG_MAX)

        return mean, log_std

    def sample_action(self,s):
        mean, log_std = self.forward(s)
        normal = torch.distributions.Normal(0,1)
        x = normal.rsample()
        y = torch.tanh(x)
        a = y*self.action_scale + self.action_bias
        log_pi = normal.log_prob(x)

        # Enforcing action bound
        log_pi -= torch.log(self.action_scale*(1-y.pow(2)) + self.epsilon)
        log_pi = log_pi.sum(1,keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return a, log_pi, mean