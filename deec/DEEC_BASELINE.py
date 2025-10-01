import numpy as np
import random
import matplotlib.pyplot as plt

from deec.utils import distance, measure_metrics
from deec.L5_comm import radio_comm
from deec.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, N_HALF, P_OPT

def run_deec():
    np.random.seed(42)
    random.seed(42)

    # Logging arrays
    avg_energy_history = []
    pdr_history = []
    alive_nodes_history = []
    num_ch_history = []
    reward_history = []          # DEEC doesn’t use RL
    epsilon_history = []
    throughput_history = []
    pdr_percent_history = []
    first_dead_round = None
    half_dead_round = None
    last_dead_round = None

    # === Environment initialization ===
    nodes_pos = [(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)]
    nodes_energy = np.array([INIT_ENERGY]*N_NODES)
    base_station = (AREA_SIZE/2, AREA_SIZE/2)  # center

    # Loop for rounds
    for rnd in range(1, ROUNDS+1):
        alive_nodes = np.sum(nodes_energy > 0)
        if alive_nodes == 0:
            print(f"[DEEC] All nodes died at round {rnd}")
            break

        # === Step 1: Compute average residual energy ===
        avg_energy = np.mean(nodes_energy[nodes_energy > 0])

        # === Step 2: CH Selection (DEEC rule) ===
        chs = []
        for i in range(N_NODES):
            if nodes_energy[i] <= 0: 
                continue
            Pi = P_OPT * (nodes_energy[i] / avg_energy)   # DEEC probability
            if random.random() < Pi:
                chs.append(i)

        if len(chs) == 0:
            # fallback: pick highest energy node alive
            alive_nodes_idx = [i for i in range(N_NODES) if nodes_energy[i] > 0]
            if alive_nodes_idx:
                chs.append(max(alive_nodes_idx, key=lambda n: nodes_energy[n]))

        # === Step 3: Cluster Formation ===
        clusters = [[] for _ in range(len(chs))]
        for i in range(N_NODES):
            if nodes_energy[i] <= 0: continue
            nearest_ch = min(range(len(chs)), key=lambda j: distance(nodes_pos[i], nodes_pos[chs[j]]))
            clusters[nearest_ch].append(i)

        # === Step 4: Communication & Energy Update ===
        nodes_energy, successful_packets, total_packets = radio_comm(
            clusters, chs, nodes_energy, nodes_pos, base_station, tx_power_factor=1.0,
            successful_packets=0, total_packets=0
        )

        nodes_energy = np.maximum(nodes_energy, 0.0)

        # === Step 5: Metrics ===
        avg_e_after, pdr_after, alive_after, cluster_sizes = measure_metrics(
            nodes_energy, clusters, chs, successful_packets, total_packets
        )

        if first_dead_round is None and alive_after < N_NODES:
            first_dead_round = rnd
        if half_dead_round is None and alive_after <= N_HALF:
            half_dead_round = rnd
        if alive_after == 0 and last_dead_round is None:
            last_dead_round = rnd

        avg_energy_history.append(avg_e_after)
        pdr_history.append(pdr_after)
        alive_nodes_history.append(alive_after)
        num_ch_history.append(len(chs))
        reward_history.append(0.0)
        epsilon_history.append(0.0)
        throughput_history.append(successful_packets)
        pdr_percent_history.append(100.0 * successful_packets / (total_packets+1e-12))

        plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

    return (
        avg_energy_history, pdr_history, alive_nodes_history, num_ch_history,
        reward_history, epsilon_history, throughput_history, pdr_percent_history,
        first_dead_round, half_dead_round, last_dead_round
    )


def plot_cluster(rnd, clusters, nodes_pos, chs, base_station):
    # Cluster plot every 400 rounds
    if rnd % 400 == 0:
        plt.figure(figsize=(6,6))
        for cidx, members in enumerate(clusters):
            x = [nodes_pos[n][0] for n in members]
            y = [nodes_pos[n][1] for n in members]
            plt.scatter(x, y, s=20, label=f'Cluster {cidx}' if cidx<5 else None)
            ch_pos = nodes_pos[chs[cidx]]
            for n in members:
                plt.plot([nodes_pos[n][0], ch_pos[0]], [nodes_pos[n][1], ch_pos[1]], 'gray', alpha=0.5)
        ch_x = [nodes_pos[ch][0] for ch in chs]
        ch_y = [nodes_pos[ch][1] for ch in chs]
        plt.scatter(ch_x, ch_y, c='red', s=80, marker='*', label='CH')
        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'DEEC Cluster plot at round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()