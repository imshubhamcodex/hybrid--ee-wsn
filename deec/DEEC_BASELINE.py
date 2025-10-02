import numpy as np
import random
import matplotlib.pyplot as plt

from deec.utils import distance, measure_metrics
from deec.L5_comm import radio_comm
from deec.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, N_HALF, P_OPT

def select_CHs_deec(nodes_energy, alive_nodes, round_num, p_opt=P_OPT):
    """
    Improved DEEC CH selection to enhance network lifetime.
    """
    E_avg = np.sum(nodes_energy) / max(np.sum(alive_nodes), 1)
    is_CH = np.zeros(N_NODES, dtype=bool)

    for i in range(N_NODES):
        if not alive_nodes[i]:
            continue

        Pi = p_opt * (nodes_energy[i] / E_avg) if E_avg > 0 else 0
        if Pi <= 0: 
            continue

        # DEECP round-based threshold
        rounds_until_next_CH = int(1 / Pi) if Pi > 0 else 1
        denominator = 1 - Pi * (round_num % rounds_until_next_CH)
        threshold = Pi / denominator if denominator != 0 else 0

        if np.random.rand() < threshold:
            is_CH[i] = True

    # fallback: if no CHs selected, pick highest-energy alive node
    if not is_CH.any():
        alive_idx = np.where(alive_nodes)[0]
        if len(alive_idx) > 0:
            is_CH[alive_idx[np.argmax(nodes_energy[alive_idx])]] = True

    return is_CH

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
        alive_nodes = nodes_energy > 0

        # === Step 1: Compute average residual energy ===
        alive_energy = nodes_energy[alive_nodes]
        avg_energy = np.mean(alive_energy) if alive_energy.size > 0 else 0

        # === Step 2: CH Selection (Improved DEEC rule) ===
        is_CH = select_CHs_deec(nodes_energy, alive_nodes, rnd)
        chs = np.where(is_CH)[0].tolist()

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

        # plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

        if alive_after == 0:
            print(f"[DEEC] All nodes died at round {rnd}")
            break

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