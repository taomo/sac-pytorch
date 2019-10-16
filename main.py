import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from helper import NormalizedActions, plot_reward, TimeFeatureWrapper
from models import SoftActorCritic
import hyp

def main():
    # Initialize environment and agent
    env = TimeFeatureWrapper(NormalizedActions(gym.make('HopperBulletEnv-v0')))
    agent = SoftActorCritic(env.observation_space, env.action_space)
    # scheduler = StepLR(agent.policy_network_opt,step_size=10,gamma=0.9)
    i = 0
    writer = SummaryWriter()

    while i < hyp.MAX_FRAMES:
        state = env.reset()
        episode_reward = 0
        
        for _ in range(hyp.MAX_STEPS):
            if i > hyp.EXPLORATION_TIME:
                action = agent.get_action(state).detach().numpy()
                next_state, reward, done, _ = env.step(action)
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
            
            if i >= 30000:
                env.render()
            agent.replay_memory.push((state,action,reward,next_state,done))            
            episode_reward += reward
            state = next_state
            i += 1
            
            if agent.replay_memory.get_len() > hyp.BATCH_SIZE: 
                policy_loss = agent.update_params()
                if i % 1000 == 0:
                    writer.add_scalar('policy_loss', policy_loss, i)

            if i % 1000 == 0:
                print("Frame: {}, reward: {}, policy_loss: {}".format(
                    i, episode_reward, policy_loss
                ))

            if done:
                break

        # scheduler.step()
        writer.add_scalar('episode_reward',episode_reward,i)

    env.close()
    writer.close()

if __name__ == '__main__':
    main()