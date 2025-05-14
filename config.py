import logging
from logging.handlers import RotatingFileHandler
import torch

# Logging setup
log_file = "poker_ai.log"
handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Add a CPU_ONLY flag to control device selection
CPU_ONLY = False  # Set to True to use CPU only and avoid DirectML warning

# Modify the device selection logic to respect CPU_ONLY flag
if torch and torch.cuda.is_available() and not CPU_ONLY:
    device = torch.device("cuda")
    logger.info("Using CUDA device.")
else:
    try:
        import torch_directml

        if not CPU_ONLY:
            device = torch_directml.device()
            logger.info(
                "CUDA not found. DirectML device found and compatible. Using DirectML."
            )
        else:
            device = torch.device("cpu")
            logger.info("CPU mode enforced. Using CPU device.")
    except ImportError:
        device = torch.device("cpu")
        logger.info("CUDA and DirectML not found. Using CPU device.")

print(f"Using device: {device}", flush=True)  # Keep print for immediate feedback


class Config:
    INITIAL_STACK = 1000
    SMALL_BLIND = 10
    BIG_BLIND = 20
    BATCH_SIZE = 512
    REPLAY_BUFFER_CAPACITY = 50000
    NUM_SIMULATIONS = 1
    NUM_TRAINING_STEPS = 5
    META_TOURNAMENT_GAMES = 200
    MAX_BETTING_ROUNDS = 10
    TRAINING_HANDS_BR = 200
    STATE_SIZE = 19  # Updated: added min_raise_norm
    ACTION_SIZE = 3  # FOLD, CALL, RAISE (continuous)
    GAMMA = 0.99
    SEED = 42
    LR = 0.001
    T_MAX = 1000
    ACTOR_HIDDEN_SIZE = 512
    CRITIC_HIDDEN_SIZE = 512
    RESIDUAL_DROPOUT = 0.1
    NUM_RES_BLOCKS = 6
    OPPONENT_DECAY = 0.9
