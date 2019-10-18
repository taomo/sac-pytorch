import gym
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

from models import SoftActorCritic
import hyp

def main():
    # Initialize environment and agent
    env = gym.make('HalfCheetahBulletEnv-v0')

    agent = SoftActorCritic(env.observation_space, env.action_space)
    i = 0
    ep = 1
    writer = SummaryWriter()

    while ep >= 1:
        episode_reward = 0
        state = env.reset()
        done = False
        j = 0
        
        while not done:
            if i > hyp.EXPLORATION_TIME:
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()
            
            if agent.replay_memory.get_len() > hyp.BATCH_SIZE: 
                q1_loss, q2_loss, policy_loss, alpha_loss = agent.update_params()
                    
                writer.add_scalar('loss/q1_loss', q1_loss, i)
                writer.add_scalar('loss/q2_loss', q2_loss, i)
                writer.add_scalar('loss/policy_loss', policy_loss, i)
                writer.add_scalar('loss/alpha_loss',alpha_loss,i)
                writer.add_scalar('loss/alpha',agent.alpha,i)

            next_state, reward, done, _ = env.step(action)
            i += 1
            j += 1
            episode_reward += reward

            ndone = 1 if j == env._max_episode_steps else float(not done)
            agent.replay_memory.push((state,action,reward,next_state,ndone))
            state = next_state
        
        if i > hyp.MAX_STEPS:
            break

        writer.add_scalar('reward/episode_reward', episode_reward, ep)
        if ep % 100 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(ep, i, j, episode_reward))
        ep += 1

    env.close()
    writer.close()

if __name__ == '__main__':
    main()