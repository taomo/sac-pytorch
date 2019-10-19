## sac-pytorch

PyTorch implementation of the Soft Actor-Critic algorithm (without separate value function network)

### Results (until now)
1. **Pendulum-v0**

![Pendulum-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/Pendulum-v0.png)

2. **LunarLanderContinuous-v2**

![LunarLanderContinuous-v2](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/LunarLander-v2.png)

3. **HopperBulletEnv-v0**

![HopperBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/HopperBulletEnv-v0.svg)

4. **HalfCheetahBulletEnv-v0**

![HalfCheetahBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/HalfCheetahBulletEnv-v0-more_epis.svg)

**Note**: Tensorboard log files for all above experiments can be found within the runs subdirectory

### Libraries used
- PyTorch
- OpenAI Gym
- PyBullet
- TensorBoard

### References
- https://github.com/araffin/rl-baselines-zoo
- https://spinningup.openai.com/en/latest/algorithms/sac.html
- https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
