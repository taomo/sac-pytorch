import torch

# Hyperparameters (For pendulum)
TAU = 0.01
EPSILON = 1e-6
H_DIM = 256
LR = 3e-4
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 256
ALPHA = 0.01
GAMMA = 0.98
ENTROPY_TUNING = True
TARGET_SOFT_UPDATE_INTERVAL = 1
TARGET_COPY_INTERVAL = 1000
MAX_STEPS = 2000000
EXPLORATION_TIME = 10000
MIN_LOG = -20
MAX_LOG = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')