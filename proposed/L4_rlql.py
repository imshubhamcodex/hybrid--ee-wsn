import random
import numpy as np
from proposed.config import ACTIONS, ALPHA, GAMMA, EPS_MIN, EPS_DECAY
from proposed.utils import state_from_metrics



def choose_action(state, epsilon, Q):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        qs = Q[state]
        # tie-break random among maxima
        maxv = np.max(qs)
        maxacts = [a for a,v in enumerate(qs) if abs(v-maxv) < 1e-9]
        return random.choice(maxacts)
    
def compute_reward(avg_e_before, avg_e_after, pdr_after, action_cost, baseline_pdr):
    delta_E = avg_e_after - avg_e_before
    delta_PDR = pdr_after - baseline_pdr
    # Reward weights (tunable)
    R_alpha = 10
    R_beta = 5
    R_gamma = 0.1
    reward = R_alpha * delta_E + R_beta * delta_PDR - R_gamma * action_cost
    return reward

def update_q(Q, state, action, reward, avg_e_after, cluster_sizes, pdr_after, epsilon, LOAD_BINS, PDR_BINS):
    # observe next state
    next_state = state_from_metrics(avg_e_after, cluster_sizes, pdr_after, LOAD_BINS, PDR_BINS)
    # q-update
    s = state
    a = action
    oldQ = Q[s][a]
    Q[s][a] = oldQ + ALPHA * (reward + GAMMA * np.max(Q[next_state]) - oldQ)
    # epsilon decay
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
    return Q, epsilon