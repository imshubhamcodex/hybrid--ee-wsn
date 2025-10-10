import numpy as np
import random
import matplotlib.pyplot as plt

from pso.utils import distance, measure_metrics
from pso.L5_comm import radio_comm
from pso.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, N_HALF, NUM_CLUSTERS, PACKET_SIZE_BITS

# PSO parameters
W = 0.8       # inertia weight
C1 = 1.0      # cognitive weight
C2 = 1.0      # social weight
PSO_POP = NUM_CLUSTERS  # number of particles
GA_GEN = 5    # number of GA generations per round

if ROUNDS <= 1000:
    W = 0.9
    PSO_POP = 25

def run_pso():
    np.random.seed(42)
    random.seed(42)

    # Logging arrays
    avg_energy_history = []
    pdr_history = []
    alive_nodes_history = []
    num_ch_history = []
    reward_history = []        # no RL
    epsilon_history = []       # no RL
    throughput_history = []
    pdr_percent_history = []
    first_dead_round = None
    half_dead_round = None
    last_dead_round = None

    # Environment initialization
    nodes_pos = np.array([(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)])
    nodes_energy = np.array([INIT_ENERGY] * N_NODES)
    base_station = (AREA_SIZE / 2, AREA_SIZE / 2)

    # Initialize PSO particles: each particle = indices of candidate CHs
    particles = [np.random.choice(range(N_NODES), NUM_CLUSTERS, replace=False) for _ in range(PSO_POP)]
    velocities = [np.zeros(NUM_CLUSTERS) for _ in range(PSO_POP)]  # simple 1D velocities per CH index
    pbest = particles.copy()
    pbest_scores = [float('inf')] * PSO_POP
    gbest = None
    gbest_score = float('inf')

    for rnd in range(1, ROUNDS + 1):
        # Evaluate fitness: sum of intra-cluster distances
        scores = []
        for pidx, ch_indices in enumerate(particles):
            clusters = [[] for _ in range(NUM_CLUSTERS)]
            for i in range(N_NODES):
                if nodes_energy[i] <= 0:
                    continue
                nearest = np.argmin([distance(nodes_pos[i], nodes_pos[ch]) for ch in ch_indices])
                clusters[nearest].append(i)
            # Fitness = sum of distances from members to CH
            fitness = sum(distance(nodes_pos[node], nodes_pos[ch_indices[cidx]])
                          for cidx, cluster_nodes in enumerate(clusters) for node in cluster_nodes)
            scores.append(fitness)
            # Update personal best
            if fitness < pbest_scores[pidx]:
                pbest_scores[pidx] = fitness
                pbest[pidx] = ch_indices.copy()
            # Update global best
            if fitness < gbest_score:
                gbest_score = fitness
                gbest = ch_indices.copy()

        # PSO velocity and position update
        for pidx in range(PSO_POP):
            for c in range(NUM_CLUSTERS):
                r1, r2 = random.random(), random.random()
                velocities[pidx][c] = (W * velocities[pidx][c] +
                                       C1 * r1 * (pbest[pidx][c] - particles[pidx][c]) +
                                       C2 * r2 * (gbest[c] - particles[pidx][c]))
                # Update particle (round to nearest node index)
                new_pos = int(round(particles[pidx][c] + velocities[pidx][c]))
                new_pos = max(0, min(N_NODES - 1, new_pos))
                particles[pidx][c] = new_pos

        # Apply GA crossover + mutation to top 50% particles
        top_half = sorted(range(PSO_POP), key=lambda i: scores[i])[:PSO_POP // 2]
        for i in range(0, len(top_half), 2):
            if i + 1 >= len(top_half):
                break
            parent1 = particles[top_half[i]]
            parent2 = particles[top_half[i + 1]]
            # single point crossover
            point = random.randint(1, NUM_CLUSTERS - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            # mutation: swap two CH indices
            for child in [child1, child2]:
                if random.random() < 0.1:
                    a, b = random.sample(range(NUM_CLUSTERS), 2)
                    child[a], child[b] = child[b], child[a]
            # replace worst particles
            worst1 = np.argmax(scores)
            worst2 = np.argsort(scores)[-2]
            particles[worst1] = child1
            particles[worst2] = child2

        # Choose final CHs for this round = global best
        chs = gbest.copy()

        # Build clusters for energy computation
        clusters = [[] for _ in range(NUM_CLUSTERS)]
        for i in range(N_NODES):
            if nodes_energy[i] <= 0:
                continue
            nearest = np.argmin([distance(nodes_pos[i], nodes_pos[ch]) for ch in chs])
            clusters[nearest].append(i)

        # Energy & communication update
        nodes_energy, successful_packets, total_packets = radio_comm(
            clusters, chs, nodes_energy, nodes_pos, base_station, tx_power_factor=1.0
        )
        nodes_energy = np.maximum(nodes_energy, 0.0)

        # Metrics
        avg_e_after, pdr_after, alive_after, cluster_sizes = measure_metrics(
            nodes_energy, clusters, chs, successful_packets, total_packets
        )

        # Logging life events
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
            print(f"[PSO+GA] All nodes died at round {rnd}")
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
            if cidx < len(chs):
                ch_pos = nodes_pos[int(chs[cidx])]
                for n in members:
                    plt.plot([nodes_pos[n][0], ch_pos[0]], [nodes_pos[n][1], ch_pos[1]], 'gray', alpha=0.5)

        # CHs
        if len(chs) > 0:
            ch_x = [nodes_pos[ch][0] for ch in chs]
            ch_y = [nodes_pos[ch][1] for ch in chs]
            plt.scatter(ch_x, ch_y, c='red', s=80, marker='*', label='CH')

        # Base station
        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'PSO + GA Clustering at round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
