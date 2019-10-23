import torch

# Hyperparameters
ENV = 'Hopper-v2' # 'HalfCheetah-v2'
TAU = 0.005 # 0.01
EPSILON = 1e-6
H_DIM = 256
LR = 3e-4
REPLAY_MEMORY_SIZE = 1000000
BATCH_SIZE = 256
ALPHA = 0.01
GAMMA = 0.99 # 0.98
ENTROPY_TUNING = False # True
MAX_STEPS = 2000000
EXPLORATION_TIME = 1000
MIN_LOG = -20
MAX_LOG = 2
TENSORBOARD_LOGS = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')