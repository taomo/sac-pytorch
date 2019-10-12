import numpy as np 
import torch 
import gym 

from models import SoftActorCritic
from helper import ReplayMemory

import argparse

def main():
    # interface
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',default='Hopper-v2')
    parser.add_argument('--gamma',type=float,default=0.99)
    parser.add_argument('--alpha',type=float,default=0.2)
    parser.add_argument('--tau',type=float,default=0.5)
    parser.add_argument('--target-update-interval',type=int,default=100)
    parser.add_argument('--replay_memory_capacity',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=24)
    parser.add_argument('--hidden_dim',type=int,default=100)
    parser.add_argument('--lr',type=float,default=0.1)
    parser.add_argument('--exploitation_threshold',type=int,default=20)
    parser.add_argument('--update_granularity',type=int,default=20)
    parser.add_argument('--max_steps',type=int,default=10000)

    args = parser.parse_args()

    # create environment, agent
    env = gym.make(args.env)
    agent = SoftActorCritic(env.observation_space[0],env.action_space,args)
    replay_memory = ReplayMemory(args.replay_memory_capacity)

    # Training hyperparameters
    n_total_steps = 0
    t_ups = 0
    i = 1

    while True:
        episode_reward = 0
        episode_steps = 0
        done = False
        s = env.reset()

        # train loop
        while not done:
            if args.exploitation_threshold > n_total_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(s)

            if len(replay_memory) > args.batch_size:
                for j in range(args.update_granularity):
                    qv1_loss, qv2_loss, pi_loss = agent.update_params(replay_memory,args.batch_size,t_ups)
                    t_ups += 1
                
            next_s, reward, done, _  = env.step(action)
            episode_steps += 1
            n_total_steps += 1
            episode_reward += reward

            replay_memory.append((s,action,reward,next_s))

            s = next_s

        if n_total_steps >= args.max_steps:
            break

        print("Episode: {}, total_steps: {}, episode_steps: {}, reward: {}".format(
            i, n_total_steps, episode_steps, round(episode_reward,2) 
        ))

        i += 1

    env.close()