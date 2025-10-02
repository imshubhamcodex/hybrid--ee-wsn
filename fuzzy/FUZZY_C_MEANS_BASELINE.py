import numpy as np
import random
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from fuzzy.utils import measure_metrics
from fuzzy.L5_comm import radio_comm
from fuzzy.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, N_HALF, NUM_CLUSTERS


def run_fuzzy():
    np.random.seed(42)
    random.seed(42)

    # Logging arrays
    avg_energy_history = []
    pdr_history = []
    alive_nodes_history = []
    num_ch_history = []
    reward_history = []          # no RL in fuzzy
    epsilon_history = []         # same here
    throughput_history = []
    pdr_percent_history = []
    first_dead_round = None
    half_dead_round = None
    last_dead_round = None

    # === Environment initialization ===
    nodes_pos = np.array([(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)])
    nodes_energy = np.array([INIT_ENERGY] * N_NODES)
    base_station = (AREA_SIZE / 2, AREA_SIZE / 2)

    # FCM parameters
    m = 2.0               # fuzziness coefficient
    error = 1e-5
    maxiter = 1000

    # Loop for rounds
    for rnd in range(1, ROUNDS + 1):

        # === Step 1: Fuzzy C-Means clustering ===
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=nodes_pos.T,
            c=NUM_CLUSTERS,
            m=m,
            error=error,
            maxiter=maxiter,
            init=None,
            seed=42
        )

        # Assign each node to cluster with highest membership
        labels = np.argmax(u, axis=0)

        # Build clusters
        clusters = [[] for _ in range(NUM_CLUSTERS)]
        for i, lbl in enumerate(labels):
            if nodes_energy[i] > 0:  # only alive nodes
                clusters[lbl].append(i)

        # --- Step 2: Pick CHs probabilistically based on node energy ---
        chs = [None] * NUM_CLUSTERS
        for cidx, members in enumerate(clusters):
            alive_members = [n for n in members if nodes_energy[n] > 0]
            if not alive_members:
                continue
            energies = np.array([nodes_energy[n] for n in alive_members])
            if energies.sum() == 0:
                chosen = random.choice(alive_members)
            else:
                probs = energies / energies.sum()
                chosen = np.random.choice(alive_members, p=probs)
            chs[cidx] = chosen

        # Remove empty clusters/None CHs
        clusters = [c for c in clusters if c]
        chs = [ch for ch in chs if ch is not None]

        # Edge case: if no CHs left, pick one random alive node
        if len(chs) == 0:
            alive_nodes = [i for i in range(N_NODES) if nodes_energy[i] > 0]
            if alive_nodes:
                chosen = random.choice(alive_nodes)
                chs = [chosen]
                clusters = [[chosen]]

        # === Step 3: Communication & Energy Update ===
        nodes_energy, successful_packets, total_packets = radio_comm(
            clusters, chs, nodes_energy, nodes_pos, base_station, tx_power_factor=1.0,
            successful_packets=0, total_packets=0
        )
        nodes_energy = np.maximum(nodes_energy, 0.0)

        # === Step 4: Metrics ===
        avg_e_after, pdr_after, alive_after, cluster_sizes = measure_metrics(
            nodes_energy, clusters, chs, successful_packets, total_packets
        )

        # logging life events
        if first_dead_round is None and alive_after < N_NODES:
            first_dead_round = rnd
        if half_dead_round is None and alive_after <= N_HALF:
            half_dead_round = rnd
        if alive_after == 0 and last_dead_round is None:
            last_dead_round = rnd

        # log histories
        avg_energy_history.append(avg_e_after)
        pdr_history.append(pdr_after)
        alive_nodes_history.append(alive_after)
        num_ch_history.append(len(chs))
        reward_history.append(0.0)
        epsilon_history.append(0.0)
        throughput_history.append(successful_packets)
        pdr_percent_history.append(100.0 * successful_packets / (total_packets + 1e-12))

        # plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

        if alive_after == 0:
            print(f"[FUZZY] All nodes died at round {rnd}")
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

        # Base station
        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'Fuzzy C-Means Clustering at round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
