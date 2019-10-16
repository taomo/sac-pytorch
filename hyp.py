import torch

# Hyperparameters (For pendulum)
RHO = 0.005
EPSILON = 1e-6
H_DIM = 256
LR = 1e-4
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 256
ALPHA = 0.01
GAMMA = 0.995
ENTROPY_TUNING = False
TARGET_SOFT_UPDATE_INTERVAL = 1
TARGET_COPY_INTERVAL = 1000
MAX_FRAMES = 2000000
MAX_STEPS = 5000
EXPLORATION_TIME = 100
MIN_LOG = -20
MAX_LOG = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')