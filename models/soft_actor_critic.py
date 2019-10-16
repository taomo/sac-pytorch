import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

from .q_network import QNetwork
from .policy_network import PolicyNetwork
from helper import ReplayMemory, copy_params, soft_update_params
import hyp

class SoftActorCritic(object):
    def __init__(self,observation_space,action_space):
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.alpha = hyp.ALPHA
        self.entropy_tuning = hyp.ENTROPY_TUNING

        self.q_network_1 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.q_network_2 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.target_q_network_1 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.target_q_network_2 = QNetwork(self.s_dim,self.a_dim,hyp.H_DIM).to(hyp.device)
        self.policy_network = PolicyNetwork(self.s_dim, self.a_dim, hyp.H_DIM).to(hyp.device)

        copy_params(self.target_q_network_1, self.q_network_1)
        copy_params(self.target_q_network_2, self.q_network_2)
        
        self.q_network_1_opt = opt.Adam(self.q_network_1.parameters(),hyp.LR)
        self.q_network_2_opt = opt.Adam(self.q_network_2.parameters(),hyp.LR)
        
        if self.entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(hyp.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=hyp.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=hyp.LR)
        
        self.policy_network_opt = opt.Adam(self.policy_network.parameters(),hyp.LR)

        self.replay_memory = ReplayMemory(hyp.REPLAY_MEMORY_SIZE)

    def get_action(self, s):
        return self.policy_network.sample_action(torch.FloatTensor(s).to(hyp.device))[0].cpu()

    def update_params(self):
        state, action, reward, next_state, done = self.replay_memory.sample(hyp.BATCH_SIZE)
        
        # make sure all are torch tensors
        state = torch.FloatTensor(state).to(hyp.device)
        action = torch.FloatTensor(action).to(hyp.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(hyp.device)
        next_state = torch.FloatTensor(next_state).to(hyp.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(hyp.device)

        # compute targets
        next_action, next_log_pi = self.policy_network.sample_action(next_state)
        next_target_q1 = self.target_q_network_1(next_state,next_action)
        next_target_q2 = self.target_q_network_2(next_state,next_action)
        next_target_q = torch.min(next_target_q1,next_target_q2) - self.alpha*next_log_pi
        next_q = reward + hyp.GAMMA*(1 - done)*next_target_q

        # compute losses
        q1 = self.q_network_1(state,action)
        q2 = self.q_network_2(state,action)
        # print(next_q.shape, q1.shape, q2.shape)

        q1_loss = F.mse_loss(q1,next_q)
        q2_loss = F.mse_loss(q2,next_q)
        
        pi, log_pi = self.policy_network.sample_action(state)
        q1_pi = self.q_network_1(state,pi)
        q2_pi = self.q_network_2(state,pi)
        min_q_pi = torch.min(q1_pi,q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # alpha loss
        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # gradient descent
        self.q_network_1_opt.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.q_network_1_opt.step()

        self.q_network_2_opt.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.q_network_2_opt.step()

        self.policy_network_opt.zero_grad()
        policy_loss.backward()
        self.policy_network_opt.step()

        # update target network params
        soft_update_params(self.target_q_network_1,self.q_network_1)
        soft_update_params(self.target_q_network_2,self.q_network_2)

        return policy_loss