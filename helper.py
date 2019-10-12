from collections import deque
import random

import torch.nn as nn

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([])
    
    def append(self,x):
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append(x)

    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

    def get_len(self):
        return len(self.memory)

def init_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias,0)

def copy_params(target, source):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(sparam.data)

def update_target(target, source, tau):
    for tparam, sparam in zip(target.parameters(), source.parameters()):
        tparam.data.copy_(tparam.data * (1.0 - tau) + sparam.data * tau)