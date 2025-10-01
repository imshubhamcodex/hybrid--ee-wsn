import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

from proposed.config import *
from proposed.L1_deec import deec_select
from proposed.L2_eeka import ee_ka_select
from proposed.L3_kmean import kmeans
from proposed.L4_rlql import choose_action, compute_reward, update_q
from proposed.L5_comm import radio_comm
from proposed.utils import distance, build_bins, state_from_metrics, measure_metrics

np.random.seed(42)
random.seed(42)

def run_proposed():
    # Logging arrays
    avg_energy_history = []
    pdr_history = []
    alive_nodes_history = []
    num_ch_history = []
    reward_history = []
    epsilon_history = []
    throughput_history = []
    pdr_percent_history = []
    first_dead_round = None
    half_dead_round = None
    last_dead_round = None

    E_BINS, LOAD_BINS, PDR_BINS = build_bins() 

    # Q-table
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    epsilon = EPSILON

    # === Environment initialization ===
    nodes_pos = [(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)]
    nodes_energy = np.array([INIT_ENERGY]*N_NODES)
    base_station = (AREA_SIZE/2, AREA_SIZE/2)  # center

    # state for tx power adjustments (simple factor)
    tx_power_factor = 1.0

    for rnd in range(1, ROUNDS+1):
        # L1: DEEC CH Candidate Selection
        candidates = deec_select(nodes_energy, P_OPT)

        # L2: EEKA Exact Best NUM_CLUSTERS Cluster Heads
        selected_chs = ee_ka_select(candidates, nodes_energy, nodes_pos, NUM_CLUSTERS)
        init_centroids = [nodes_pos[i] for i in selected_chs]

        # L3: K-Means Define Compact Clustering
        clusters, centroids = kmeans(nodes_pos, NUM_CLUSTERS, init_points=init_centroids)
        
        # Hybrid CH Assignment
        final_chs = []

        for cidx, cluster_nodes in enumerate(clusters):
            if len(cluster_nodes) == 0:
                # Fallback: Pick highest-energy overall
                final_chs.append(int(np.argmax(nodes_energy)))
                continue

            centroid = centroids[cidx]
            
            # Find EEKA CHs that are inside this cluster
            ee_ka_in_cluster = [ch for ch in selected_chs if ch in cluster_nodes]
            
            if ee_ka_in_cluster:
                # Pick EEKA CH closest to cluster centroid
                nearest_ch = min(ee_ka_in_cluster, key=lambda ch: distance(nodes_pos[ch], centroid))
                final_chs.append(nearest_ch)
            else:
                # Fallback: Pick highest-energy node in the cluster
                highest_energy_node = max(cluster_nodes, key=lambda n: nodes_energy[n])
                final_chs.append(highest_energy_node)

        # Final CH list
        chs = final_chs

        # L4: Q-Learning based Local Adjustments

        # L4-1: RL observes state
        avg_e_before = np.mean(nodes_energy)
        baseline_pdr = 1.0
        state = state_from_metrics(avg_e_before, [len(c) for c in clusters], baseline_pdr, LOAD_BINS, PDR_BINS)

        # L4-2: RL action selection
        action = choose_action(state, epsilon, Q)
        max_cluster = max([len(c) for c in clusters]) if clusters else 0
        mean_cluster = np.mean([len(c) for c in clusters]) if clusters else 0
        allow_reassign = (max_cluster > 1.5 * mean_cluster)
        if action == A_REASSIGN_FEW and not allow_reassign:
            action = A_NOOP

        # L4-3: RL Execute action (local tweaks)
        action_cost = 0.0
        if action == A_NOOP:
            pass
        elif action == A_REASSIGN_FEW:
            # move up to m nodes from largest CH to nearest underloaded CH
            largest_idx = int(np.argmax([len(c) for c in clusters]))
            m = max(1, int(0.05 * len(clusters[largest_idx])))

            
            members = clusters[largest_idx].copy()

            # sort members by distance to CH descending (move farthest ones)
            members.sort(key=lambda n: distance(nodes_pos[n], nodes_pos[chs[largest_idx]]), reverse=True)
            moved = 0

            for node in members:
                if moved >= m: break
                # find nearest underloaded CH (size < mean)
                candidates_ch = [i for i in range(len(clusters)) if len(clusters[i]) < mean_cluster]
                if not candidates_ch: break
                # pick the one with minimum distance
                best = min(candidates_ch, key=lambda cidx: distance(nodes_pos[node], nodes_pos[chs[cidx]]))
                # reassign
                clusters[largest_idx].remove(node)
                clusters[best].append(node)
                moved += 1
                action_cost += 0.2  # small comm cost
        elif action == A_SWITCH_CH:
            # pick an overloaded CH and handover role to highest-energy CM in that cluster
            largest_idx = int(np.argmax([len(c) for c in clusters]))
            if len(clusters[largest_idx])>1:
                # choose candidate replacement (highest energy member)
                candidate = max([n for n in clusters[largest_idx] if n!=chs[largest_idx]], key=lambda x: nodes_energy[x])
                # swap CH role
                old_ch = chs[largest_idx]
                chs[largest_idx] = candidate
                action_cost += 0.5
        elif action == A_REDUCE_TX:
            tx_power_factor = max(0.5, tx_power_factor * 0.9)
            action_cost += 0.05
        elif action == A_INCREASE_TX:
            tx_power_factor = min(1.5, tx_power_factor * 1.1)
            action_cost += 0.05

        # L5: Communication and Energy Update
        nodes_energy, successful_packets, total_packets = radio_comm(clusters, chs, nodes_energy, nodes_pos, base_station, tx_power_factor, successful_packets=0, total_packets=0)

        # clamp negative energies to 0
        nodes_energy = np.maximum(nodes_energy, 0.0)

        # L4-4: Compute Reward
        avg_e_after, pdr_after, alive_after, cluster_sizes = measure_metrics(nodes_energy, clusters, chs, successful_packets, total_packets)
        reward = compute_reward(avg_e_before, avg_e_after, pdr_after, action_cost, baseline_pdr)
        # big negative if first node just died
        if any((nodes_energy<=0) & (nodes_energy+1e-9>0)):  # simplistic check
            reward -= 5
        
        # L4-5: Q-Update
        Q, epsilon = update_q(Q, state, action, reward, avg_e_after, cluster_sizes, pdr_after, epsilon, LOAD_BINS, PDR_BINS)

        # logging
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
        reward_history.append(reward)
        epsilon_history.append(epsilon)
        throughput_history.append(successful_packets)
        pdr_percent_history.append(100.0 * successful_packets / (total_packets+1e-12))

        # plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

        # Termination if all nodes died
        if alive_after == 0:
            print(f"[RL_PROPOSED] All nodes died at round {rnd}")
            break

    return (avg_energy_history, pdr_history, alive_nodes_history, num_ch_history, 
            reward_history, epsilon_history, throughput_history, pdr_percent_history, 
            first_dead_round, half_dead_round, last_dead_round)


def plot_cluster(rnd, clusters, nodes_pos, chs, base_station):
    # Cluster plot every 1000 rounds 
    if rnd % 1000 == 0:
        plt.figure(figsize=(6,6))
        for cidx, members in enumerate(clusters):
            x = [nodes_pos[n][0] for n in members]
            y = [nodes_pos[n][1] for n in members]
            plt.scatter(x, y, s=20, label=f'Cluster {cidx}' if cidx<5 else None)  # label only first few
            # Draw links
            ch_pos = nodes_pos[chs[cidx]]
            for n in members:
                plt.plot([nodes_pos[n][0], ch_pos[0]], [nodes_pos[n][1], ch_pos[1]], 'gray', alpha=0.5)
        # Plot CHs
        ch_x = [nodes_pos[ch][0] for ch in chs]
        ch_y = [nodes_pos[ch][1] for ch in chs]
        plt.scatter(ch_x, ch_y, c='red', s=80, marker='*', label='CH')
        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'Cluster plot at round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()