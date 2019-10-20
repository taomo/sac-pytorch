import torch

# Hyperparameters
TAU = 0.005
EPSILON = 1e-6
H_DIM = 256
LR = 3e-4
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 256
ALPHA = 0.2
GAMMA = 0.99
ENTROPY_TUNING = False
TARGET_SOFT_UPDATE_INTERVAL = 1
TARGET_COPY_INTERVAL = 1000
MAX_STEPS = 100000
EXPLORATION_TIME = 1000
MIN_LOG = -20
MAX_LOG = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')