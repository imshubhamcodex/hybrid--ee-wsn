import numpy as np
import random
import matplotlib.pyplot as plt

from aco.utils import measure_metrics
from aco.L5_comm import radio_comm
from aco.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, N_HALF, NUM_CLUSTERS

# ACO parameters
ALPHA = 1.0      # pheromone importance
BETA = 2.0       # heuristic importance
RHO = 0.5        # pheromone evaporation
Q = 100          # pheromone constant
ANTS = NUM_CLUSTERS        # number of ants

def run_aco():
    np.random.seed(42)
    random.seed(42)

    # Logging arrays
    avg_energy_history, pdr_history, alive_nodes_history = [], [], []
    num_ch_history, reward_history, epsilon_history = [], [], []
    throughput_history, pdr_percent_history = [], []
    first_dead_round = half_dead_round = last_dead_round = None

    # Environment setup
    nodes_pos = np.array([(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)])
    nodes_energy = np.array([INIT_ENERGY] * N_NODES)
    base_station = (AREA_SIZE / 2, AREA_SIZE / 2)

    # Initialize pheromone and heuristic info
    pheromone = np.ones(N_NODES)
    heuristic = np.zeros(N_NODES)

    for rnd in range(1, ROUNDS + 1):
        alive_nodes = [i for i in range(N_NODES) if nodes_energy[i] > 0]
        if not alive_nodes:
            print(f"[ACO] All nodes died at round {rnd}")
            break

        # Update heuristic info
        for i in alive_nodes:
            d_bs = np.linalg.norm(nodes_pos[i] - base_station)
            heuristic[i] = nodes_energy[i] / (d_bs + 1e-6)

        # Each ant builds a solution
        all_solutions, fitness_scores = [], []
        for _ in range(ANTS):
            probs = (pheromone[alive_nodes] ** ALPHA) * (heuristic[alive_nodes] ** BETA)
            probs /= probs.sum()

            # Safe CH selection
            num_chs = min(NUM_CLUSTERS, len(alive_nodes))
            if len(alive_nodes) <= NUM_CLUSTERS:
                ch_indices = alive_nodes
            else:
                ch_indices = np.random.choice(alive_nodes, size=num_chs, replace=False, p=probs[:len(alive_nodes)])

            # Assign nodes to nearest CH
            clusters = [[] for _ in range(num_chs)]
            for i in alive_nodes:
                dists = np.linalg.norm(nodes_pos[ch_indices] - nodes_pos[i], axis=1)
                closest = np.argmin(dists)
                clusters[closest].append(i)

            # Evaluate solution
            total_dist = sum(np.min(np.linalg.norm(nodes_pos[ch_indices] - nodes_pos[i], axis=1)) for i in alive_nodes)
            energy_term = np.mean(nodes_energy[alive_nodes])
            fitness = total_dist / (energy_term + 1e-6)

            all_solutions.append((clusters, ch_indices))
            fitness_scores.append(fitness)

        # Choose best ant
        best_idx = np.argmin(fitness_scores)
        best_clusters, best_chs = all_solutions[best_idx]

        # Remove empty clusters & corresponding CHs
        valid_clusters, valid_chs = [], []
        for ci, cluster in enumerate(best_clusters):
            if cluster:
                valid_clusters.append(cluster)
                if ci < len(best_chs):
                    valid_chs.append(best_chs[ci])

        # Edge case: if no CHs left
        if not valid_chs:
            chosen = random.choice([i for i in range(N_NODES) if nodes_energy[i] > 0])
            valid_chs = [chosen]
            valid_clusters = [[chosen]]

        best_clusters, best_chs = valid_clusters, valid_chs

        # Update pheromone
        pheromone *= (1 - RHO)
        for ch in best_chs:
            pheromone[ch] += Q / (fitness_scores[best_idx] + 1e-6)

        # Communication
        nodes_energy, successful_packets, total_packets = radio_comm(
            best_clusters, best_chs, nodes_energy, nodes_pos, base_station,
            tx_power_factor=1.0, successful_packets=0, total_packets=0
        )
        nodes_energy = np.maximum(nodes_energy, 0.0)

        # Metrics
        avg_e_after, pdr_after, alive_after, _ = measure_metrics(
            nodes_energy, best_clusters, best_chs, successful_packets, total_packets
        )

        # Life markers
        if first_dead_round is None and alive_after < N_NODES: first_dead_round = rnd
        if half_dead_round is None and alive_after <= N_HALF: half_dead_round = rnd
        if alive_after == 0 and last_dead_round is None: last_dead_round = rnd

        # Logging
        avg_energy_history.append(avg_e_after)
        pdr_history.append(pdr_after)
        alive_nodes_history.append(alive_after)
        num_ch_history.append(len(best_chs))
        reward_history.append(0.0)
        epsilon_history.append(0.0)
        throughput_history.append(successful_packets)
        pdr_percent_history.append(100 * successful_packets / (total_packets + 1e-12))

        # plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

        if alive_after == 0:
            print(f"[ACO] All nodes died at round {rnd}")
            break

    return (
        avg_energy_history, pdr_history, alive_nodes_history, num_ch_history,
        reward_history, epsilon_history, throughput_history, pdr_percent_history,
        first_dead_round, half_dead_round, last_dead_round
    )



def plot_cluster(rnd, clusters, nodes_pos, chs, base_station):
    if rnd % 1000 == 0:
        plt.figure(figsize=(6, 6))
        for cidx, members in enumerate(clusters):
            if not members:
                continue
            x = [nodes_pos[n][0] for n in members]
            y = [nodes_pos[n][1] for n in members]
            plt.scatter(x, y, s=20, label=f'Cluster {cidx}' if cidx < 5 else None)

            # draw lines from members to CH
            if cidx < len(chs):
                ch_pos = nodes_pos[chs[cidx]]
                for n in members:
                    plt.plot([nodes_pos[n][0], ch_pos[0]], [nodes_pos[n][1], ch_pos[1]], 'gray', alpha=0.5)

        # CHs
        if chs:
            ch_x = [nodes_pos[ch][0] for ch in chs]
            ch_y = [nodes_pos[ch][1] for ch in chs]
            plt.scatter(ch_x, ch_y, c='red', s=80, marker='*', label='CH')

        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'ACO Clustering at Round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
