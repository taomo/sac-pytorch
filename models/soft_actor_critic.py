import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as opt

from helper import update_target, copy_params
from q_network import QNetwork
from policy_network import PolicyNetwork

class SoftActorCritic(object):
    def __init__(self, n_s, action_space, args):
        super(SoftActorCritic,self).__init__()

        # hyperparameters
        self.alpha = args.alpha
        self.gamma = args.gamma 
        self.tau = args.tau
        self.target_update_interval = args.target_update_interval
        self.hidden_dim = args.hidden_dim
        self.action_space = action_space

        # instantiate models
        self.Q1 = QNetwork(n_s,action_space.shape[0],self.hidden_dim)
        self.Q2 = QNetwork(n_s,action_space.shape[0],self.hidden_dim)
        self.targetQ1 = QNetwork(n_s,action_space.shape[0],self.hidden_dim)
        self.targetQ2 = QNetwork(n_s,action_space.shape[0],self.hidden_dim)
        self.P = PolicyNetwork(n_s,self.hidden_dim,action_space)

        # copy initial params to target networks
        copy_params(self.targetQ1,self.Q1)
        copy_params(self.targetQ2,self.Q2)

        # initialize optimizers
        self.Q1_optim = opt.Adam(self.Q1.parameters(),lr=args.lr)
        self.Q2_optim = opt.Adam(self.Q2.parameters(),lr=args.lr)
        self.P_optim = opt.Adam(self.P.parameters(), lr=args.lr)

    def get_action(self, s):
        action, _, _ = self.P.sample_action(s)

        return action

    def update_params(self, replay_memory, batch_size, t_ups):
        # sample batch from replay memory
        s_batch, a_batch, r_batch, next_s_batch = replay_memory.sample(batch_size)

        with torch.no_grad():
            next_a, next_s_log_pi, _ = self.P.sample_action(next_s_batch)
            qv1_next_target = self.targetQ1(next_s_batch, next_a)
            qv2_next_target = self.targetQ2(next_s_batch, next_a)
            min_qv_next_target = torch.min(qv1_next_target, qv2_next_target) - self.alpha*next_s_log_pi
            next_qv = r_batch + self.gamma*min_qv_next_target

        qv1 = self.Q1(s_batch, a_batch)
        qv2 = self.Q2(s_batch, a_batch)
        qv1_loss = F.mse_loss(qv1, next_qv)
        qv2_loss = F.mse_loss(qv2, next_qv)

        pi, log_pi, _ = self.P.sample(s_batch)
        qv1_pi = self.Q1(s_batch,pi)
        qv2_pi = self.Q2(s_batch,pi)
        min_qv_pi = torch.min(qv1_pi, qv2_pi)

        pi_loss = ((self.alpha*log_pi) - min_qv_pi).mean()

        self.Q1_optim.zero_grad()
        qv1_loss.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        qv2_loss.backward()
        self.Q2_optim.step()

        self.P_optim.zero_grad()
        pi_loss.backward()
        self.P_optim.step()

        if t_ups % self.target_update_interval ==0:
            update_target(self.targetQ1, self.Q1, self.tau)
            update_target(self.targetQ2, self.Q2, self.tau)

        return qv1_loss.item(), qv2_loss.item(), pi_loss.item()