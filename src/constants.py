DATA = "Med"  # or "MNIST"

# --- TRAINING PARAMETERS ---
# Number of distributed clients participating in the training process.
NR_CLIENTS = 3

#
LR = 0.01

# Number of client training epochs.
CLIENT_EPOCHS = 10

# Number of times the server is supposed to perform a global
# update step by aggregating the trained models from the clients.
NR_TRAINING_ROUNDS = 20

# Batch sizes 
if DATA == "Med":
    BATCH_SIZE = 10
elif DATA == "MNIST":
    BATCH_SIZE = 128


# --- DIFFERENTIAL PRIVACY PARAMETERS ---
MAX_GRAD_NORM = 1.2
EPSILON = 5

EPSILON_TRAINING_ITERATION = EPSILON / NR_TRAINING_ROUNDS
# DELTA = 1e-5

VIRTUAL_BATCH_SIZE = 2 * BATCH_SIZE

assert VIRTUAL_BATCH_SIZE % BATCH_SIZE == 0  # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
N_ACCUMULATION_STEPS = int(VIRTUAL_BATCH_SIZE / BATCH_SIZE)
