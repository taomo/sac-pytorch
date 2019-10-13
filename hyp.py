import torch

# Hyperparameters (For pendulum)
RHO = 0.005
EPSILON = 1e-6
H_DIM = 256
LR = 3e-4
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 128
ALPHA = 0.2
GAMMA = 0.99
TARGET_SOFT_UPDATE_INTERVAL = 1
TARGET_COPY_INTERVAL = 1000
MAX_FRAMES = 40000
MAX_STEPS = 500
EXPLORATION_TIME = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')