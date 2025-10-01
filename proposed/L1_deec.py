import random
import numpy as np

def deec_select(nodes_energy, p_opt):
    E_avg = np.mean(nodes_energy)
    candidates = []
    for i, Ei in enumerate(nodes_energy):
        # probability proportional to Ei/E_avg * p_opt
        prob = (Ei / (E_avg+1e-12)) * p_opt
        if random.random() <= prob:
            candidates.append(i)
    return candidates