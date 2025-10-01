import math

# Simulation Parameters (based on paper)
AREA_SIZE = 1000.0  # meters, square area
N_NODES = 200       # number of sensor nodes
INIT_ENERGY = 2.0   # joules per node
NUM_CLUSTERS = 15
ROUNDS = 8000
PACKET_SIZE_BYTES = 512
PACKET_SIZE_BITS = PACKET_SIZE_BYTES * 8
N_HALF = N_NODES // 2

# Radio Energy Model (First-order)
E_ELEC = 50e-9  # J/bit (typical)
EPS_FS = 10e-12  # J/bit/m^2 (free space)
EPS_MP = 0.0013e-12  # J/bit/m^4 (multipath) -- small value chosen for demo
D0 = math.sqrt(EPS_FS / EPS_MP) if EPS_MP>0 else 87.7  # threshold

# DEEC Parameters
P_OPT = NUM_CLUSTERS / N_NODES  # desired CH fraction
MAX_CH = NUM_CLUSTERS

# RL Parameters (tabular Q-learning)
STATE_BINS_E = 3       # energy levels
STATE_BINS_LOAD = 3    # channel load levels
STATE_BINS_PDR = 3     # packet delivery ratio levels
EPSILON = 1.0
EPS_MIN = 0.05         # minimum epsilon
EPS_DECAY = 0.995      # epsilon decay per episode
ALPHA = 0.1            # learning rate
GAMMA = 0.95           # discount factor

# RL Actions
A_NOOP = 0          # no operation
A_REASSIGN_FEW = 1  # reassign few nodes
A_SWITCH_CH = 2     # switch cluster head
A_REDUCE_TX = 3     # reduce transmission power
A_INCREASE_TX = 4   # increase transmission power
ACTIONS = [A_NOOP, A_REASSIGN_FEW, A_SWITCH_CH, A_REDUCE_TX, A_INCREASE_TX]