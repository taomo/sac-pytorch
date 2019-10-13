import gym
from collections import deque
import random
import numpy as np 
import matplotlib.pyplot as plt

import hyp

class ReplayMemory:
    def __init__(self,size):
        self.size = size
        self.memory = deque([],maxlen=size)

    def push(self, x):
        self.memory.append(x)
    
    def sample(self, batch_size):
        batch = random.sample(self.memory,batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def get_len(self):
        return len(self.memory)

def copy_params(target_network,source_network):
    for tp, sp in zip(target_network.parameters(), source_network.parameters()):
        tp.data.copy_(sp.data)

def soft_update_params(target_network, source_network):
    for tp, sp in zip(target_network.parameters(), source_network.parameters()):
        tp.data.copy_(hyp.RHO * tp.data + (1.0-hyp.RHO)*sp.data)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
        
def plot_reward(i, rewards):
    plt.close()
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (i, rewards[-1]))
    plt.plot(rewards)
    plt.show()