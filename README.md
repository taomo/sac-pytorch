## sac-pytorch

PyTorch implementation of the Soft Actor-Critic algorithm (without separate value function network)

### Results (until now)
| Environment     | Final episode reward     | TensorBoard events file link
| ----------------| :-----------------------:|----------------------------:|
| Pendulum-v0     | -122.6                   |https://drive.google.com/open?id=19qimsoihlaivbvSxhY9VcXS7IhuGUaEx                             |
| LunarLanderContinuous-v2 | 286.49          |https://drive.google.com/open?id=1S1wz0HHBxj2cCmAOAxgMN9C61oAjQc7u                            |
|HopperBulletEnv-v0        | 912.47          |https://drive.google.com/open?id=1yrkyXH_-DJh789JZJVs0hOxOd29B2Eox                             |
|HalfCheetahBulletEnv-v0   | 1952.3            |https://drive.google.com/open?id=1JHmsxL0vIiqHCoDH_ebvCFJLR7ya0CqQ                             | 

**Note**: For hyperparameters used for training, please refer [araffin/rl-baselines-zoo](https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/sac.yml)

### Episode reward vs Episode number Plots

1. **Pendulum-v0**

![Pendulum-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/Pendulum-v0.png)

2. **LunarLanderContinuous-v2**

![LunarLanderContinuous-v2](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/LunarLander-v2.png)

3. **HopperBulletEnv-v0**

![HopperBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/HopperBulletEnv-v0.svg)

4. **HalfCheetahBulletEnv-v0**

![HalfCheetahBulletEnv-v0](https://github.com/ajaysub110/sac-pytorch/blob/master/plots/HalfCheetahBulletEnv-v0-more_epis.svg)

### Libraries used
- PyTorch
- OpenAI Gym
- PyBullet
- TensorBoard

### References
- https://arxiv.org/abs/1801.01290
- https://github.com/araffin/rl-baselines-zoo
- https://spinningup.openai.com/en/latest/algorithms/sac.html
- https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665