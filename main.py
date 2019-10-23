import gym
# import pybullet_envs
from torch.utils.tensorboard import SummaryWriter

from models import SoftActorCritic
import hyp
from helper import TimeFeatureWrapper

print(hyp.device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    # Initialize environment and agent
    env = TimeFeatureWrapper(gym.make(hyp.ENV))
    agent = SoftActorCritic(env.observation_space, env.action_space)

    i = 0
    ep = 1

    if hyp.TENSORBOARD_LOGS:
        writer = SummaryWriter() # for tensorboard logs

    while ep >= 1:
        episode_reward = 0
        state = env.reset()
        done = False
        j = 0
        
        while not done:
            # sample action
            if i > hyp.EXPLORATION_TIME:
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()
            
            if agent.replay_memory.get_len() > hyp.BATCH_SIZE: 
                # get losses
                q1_loss, q2_loss, policy_loss, alpha_loss = agent.update_params()
                    
                # write loss logs to tensorboard
                if hyp.TENSORBOARD_LOGS:
                    writer.add_scalar('loss/q1_loss', q1_loss, i)
                    writer.add_scalar('loss/q2_loss', q2_loss, i)
                    writer.add_scalar('loss/policy_loss', policy_loss, i)
                    writer.add_scalar('loss/alpha_loss',alpha_loss,i)
                    writer.add_scalar('loss/alpha',agent.alpha,i)

            # prepare transition for replay memory push
            next_state, reward, done, _ = env.step(action)
            i += 1
            j += 1
            episode_reward += reward

            agent.replay_memory.push((state,action,reward,next_state,not done))
            state = next_state
        
        if i > hyp.MAX_STEPS:
            break

        # write episode reward to tensorboard logs
        if hyp.TENSORBOARD_LOGS:
            writer.add_scalar('reward/episode_reward', episode_reward, ep)
            writer.add_scalar('lr/policy_lr', get_lr(agent.policy_network_opt),ep)

        if ep % 10 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(ep, i, j, episode_reward))
            agent.save_model(hyp.model_path)
        ep += 1

    agent.save_model(hyp.model_path)
    
    env.close()
    if hyp.TENSORBOARD_LOGS:
        writer.close()

if __name__ == '__main__':
    main()