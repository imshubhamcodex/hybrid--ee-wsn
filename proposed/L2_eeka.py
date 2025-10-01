import numpy as np
from proposed.utils import distance
from proposed.config import NUM_CLUSTERS


def ee_ka_select(candidates, nodes_energy, nodes_pos, desired_k=NUM_CLUSTERS):
    # utility Wi: prioritize residual energy and centrality (inverse avg distance to others)
    utilities = []
    for i in candidates:
        # avg distance to other nodes (proxy for centrality)
        dists = [distance(nodes_pos[i], nodes_pos[j]) for j in range(len(nodes_pos)) if j != i]
        avgd = np.mean(dists) if len(dists)>0 else 1e6
        utility = nodes_energy[i] + (1.0 / (avgd + 1e-6))
        utilities.append((utility, i))
    utilities.sort(reverse=True)  # high utility first
    selected = [i for (_, i) in utilities[:desired_k]]
    # If not enough candidates, fill from highest energy nodes
    if len(selected) < desired_k:
        remaining = [i for i in range(len(nodes_energy)) if i not in selected]
        remaining.sort(key=lambda x: nodes_energy[x], reverse=True)
        for r in remaining[:desired_k-len(selected)]:
            selected.append(r)
    return selected[:desired_k]