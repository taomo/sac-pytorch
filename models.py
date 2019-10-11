import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as opt

# Hyperparameters
LOG_MIN = -20
LOG_MAX = 2

def init_weights(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias,0)

def copy_params(target, source):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(sparam.data)

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

class PolicyNetwork(nn.Module):
    def __init__(self, n_s, n_h, n_a):
        super(PolicyNetwork,self).__init__()

        self.linear1 = nn.Linear(n_s,n_h)
        self.linear2 = nn.Linear(n_h,n_h)
        self.mean_linear = nn.Linear(n_h,n_a)
        self.log_std_linear = nn.Linear(n_h,n_a)

        self.apply(init_weights)

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
        x = normal.sample
        a = torch.tanh(mean + log_std.exp()*x)

        return a