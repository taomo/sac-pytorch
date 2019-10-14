import gym

from helper import NormalizedActions, plot_reward
from models import SoftActorCritic
import hyp

def main():
    # Initialize environment and agent
    env = NormalizedActions(gym.make('Pendulum-v0'))
    env.seed(0)
    agent = SoftActorCritic(env.observation_space, env.action_space)
    i = 0
    rewards = []

    while i < hyp.MAX_FRAMES:
        state = env.reset()
        episode_reward = 0
        
        for _ in range(hyp.MAX_STEPS):
            if i > 17000:
                env.render()
            if i > hyp.EXPLORATION_TIME:
                action = agent.get_action(state).detach().numpy()
                next_state, reward, done, _ = env.step(action)
            else:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
            
            agent.replay_memory.push((state,action,reward,next_state,done))            
            episode_reward += reward
            state = next_state
            i += 1
            
            if agent.replay_memory.get_len() > hyp.BATCH_SIZE: 
                agent.update_params()

            if i % 1000 == 0:
                plot_reward(i, rewards)

            if done:
                break

        rewards.append(episode_reward)

    env.close()

if __name__ == '__main__':
    main()