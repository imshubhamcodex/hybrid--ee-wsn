import numpy as np
import random
import matplotlib.pyplot as plt

# Reuse existing utilities and comm functions from your pso module (common util)
from pso.utils import distance, measure_metrics
from pso.L5_comm import radio_comm
from pso.config import N_NODES, AREA_SIZE, INIT_ENERGY, ROUNDS, NUM_CLUSTERS, N_HALF

# GA hyperparameters
POP_SIZE = 20
GENS = 20
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
ELITISM = 2  # number of top individuals preserved each generation


def init_population(alive_nodes):
    """
    Population: each individual is a list of NUM_CLUSTERS node indices (candidate CHs).
    Allow repeats initially but selection and evaluation will prefer diverse & alive CHs.
    """
    pop = []
    for _ in range(POP_SIZE):
        individual = []
        # sample unique CHs if possible
        if len(alive_nodes) >= NUM_CLUSTERS:
            individual = random.sample(alive_nodes, NUM_CLUSTERS)
        else:
            # not enough alive nodes: fill with alive nodes (repeats permitted)
            while len(individual) < NUM_CLUSTERS:
                individual.append(random.choice(alive_nodes) if alive_nodes else random.randrange(N_NODES))
        pop.append(individual)
    return pop


def fitness(individual, nodes_pos, nodes_energy):
    """
    Lower is better.
    Objective: weighted sum of (total distance to nearest CH) and (inverse of average residual energy of chosen CHs).
    The idea: minimize intra-cluster distance, and prefer CHs with higher remaining energy.
    """
    # If any CH dead -> large penalty
    penalty = 0.0
    ch_positions = []
    for ch in individual:
        if nodes_energy[ch] <= 0:
            penalty += 1e6  # big penalty for dead CH
        ch_positions.append(nodes_pos[ch])

    ch_positions = np.array(ch_positions)

    # total distance: each alive node to nearest CH
    total_dist = 0.0
    alive_mask = nodes_energy > 0
    for idx, pos in enumerate(nodes_pos):
        if not alive_mask[idx]:
            continue
        dists = np.linalg.norm(ch_positions - pos, axis=1)
        total_dist += np.min(dists)

    # energy term: prefer CHs that have higher energy
    ch_energies = [nodes_energy[ch] for ch in individual]
    avg_ch_energy = np.mean(ch_energies) if len(ch_energies) > 0 else 1e-6

    # combine: distance normalized by average energy
    score = total_dist / (avg_ch_energy + 1e-9) + penalty
    return score


def tournament_selection(pop, scores):
    best = None
    for _ in range(TOURNAMENT_K):
        i = random.randrange(len(pop))
        if best is None or scores[i] < scores[best]:
            best = i
    return pop[best][:]  # return a copy


def one_point_crossover(a, b):
    if random.random() > CROSSOVER_RATE:
        return a[:], b[:]
    point = random.randint(1, NUM_CLUSTERS - 1)
    child1 = a[:point] + [g for g in b[point:]]
    child2 = b[:point] + [g for g in a[point:]]
    # ensure length
    child1 = child1[:NUM_CLUSTERS]
    child2 = child2[:NUM_CLUSTERS]
    return child1, child2


def mutate(ind, alive_nodes):
    if random.random() > MUTATION_RATE:
        return ind
    idx = random.randrange(NUM_CLUSTERS)
    if alive_nodes:
        ind[idx] = random.choice(alive_nodes)
    else:
        ind[idx] = random.randrange(N_NODES)
    return ind


def individual_to_clusters_and_chs(individual, nodes_pos, nodes_energy):
    """Assign each alive node to the nearest CH in the individual and derive CH indices used."""
    # make unique CH list preserving order
    chs = []
    for ch in individual:
        if ch not in chs and nodes_energy[ch] > 0:
            chs.append(ch)
    # if no valid CHs, pick one alive node as CH
    alive_nodes = [i for i in range(N_NODES) if nodes_energy[i] > 0]
    if not chs and alive_nodes:
        chs = [random.choice(alive_nodes)]

    clusters = [[] for _ in range(len(chs))]
    for i in range(N_NODES):
        if nodes_energy[i] <= 0:
            continue
        # find nearest ch position
        dists = [np.linalg.norm(nodes_pos[i] - nodes_pos[ch]) for ch in chs]
        argmin = int(np.argmin(dists))
        clusters[argmin].append(i)

    return clusters, chs


def run_pso():
    np.random.seed(42)
    random.seed(42)

    # Logging arrays (same return structure as other methods)
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

    # Environment initialization
    nodes_pos = np.array([(random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for _ in range(N_NODES)])
    nodes_energy = np.array([INIT_ENERGY] * N_NODES)
    base_station = np.array([AREA_SIZE / 2, AREA_SIZE / 2])

    for rnd in range(1, ROUNDS + 1):
        alive_nodes = [i for i in range(N_NODES) if nodes_energy[i] > 0]
        if not alive_nodes:
            print(f"[GA] No alive nodes at start of round {rnd}")
            break

        # Initialize population
        pop = init_population(alive_nodes)

        # evolve
        for gen in range(GENS):
            scores = [fitness(ind, nodes_pos, nodes_energy) for ind in pop]
            # elitism: keep best ELITISM individuals
            sorted_idx = np.argsort(scores)
            new_pop = [pop[i] for i in sorted_idx[:ELITISM]]

            # create rest via selection, crossover, mutation
            while len(new_pop) < POP_SIZE:
                parent1 = tournament_selection(pop, scores)
                parent2 = tournament_selection(pop, scores)
                child1, child2 = one_point_crossover(parent1, parent2)
                child1 = mutate(child1, alive_nodes)
                child2 = mutate(child2, alive_nodes)
                new_pop.append(child1)
                if len(new_pop) < POP_SIZE:
                    new_pop.append(child2)
            pop = new_pop

        # after GENS, pick best individual
        final_scores = [fitness(ind, nodes_pos, nodes_energy) for ind in pop]
        best_idx = int(np.argmin(final_scores))
        best_ind = pop[best_idx]

        # build clusters & CHs from best individual
        clusters, chs = individual_to_clusters_and_chs(best_ind, nodes_pos, nodes_energy)

        # Ensure cluster/ch sizes are aligned (safety)
        clusters = [c for c in clusters if c]
        chs = [chs[i] for i in range(len(chs)) if i < len(clusters)]
        if len(chs) != len(clusters):
            min_len = min(len(chs), len(clusters))
            clusters = clusters[:min_len]
            chs = chs[:min_len]

        # Communication phase
        nodes_energy, successful_packets, total_packets = radio_comm(
            clusters, chs, nodes_energy, nodes_pos, base_station,
            tx_power_factor=1.0, successful_packets=0, total_packets=0
        )
        nodes_energy = np.maximum(nodes_energy, 0.0)

        # Metrics
        avg_e_after, pdr_after, alive_after, _ = measure_metrics(
            nodes_energy, clusters, chs, successful_packets, total_packets
        )

        # Life markers
        if first_dead_round is None and alive_after < N_NODES:
            first_dead_round = rnd
        if half_dead_round is None and alive_after <= N_HALF:
            half_dead_round = rnd
        if alive_after == 0 and last_dead_round is None:
            last_dead_round = rnd

        # Logs
        avg_energy_history.append(avg_e_after)
        pdr_history.append(pdr_after)
        alive_nodes_history.append(alive_after)
        num_ch_history.append(len(chs))
        reward_history.append(0.0)
        epsilon_history.append(0.0)
        throughput_history.append(successful_packets)
        pdr_percent_history.append(100 * successful_packets / (total_packets + 1e-12))

        # optional visualization every X rounds (disabled by default - set modulo if you want)
        # plot_cluster(rnd, clusters, nodes_pos, chs, base_station)

        if alive_after == 0:
            print(f"[PSO] All nodes died at round {rnd}")
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
                ch_pos = nodes_pos[chs[cidx]]
                for n in members:
                    plt.plot([nodes_pos[n][0], ch_pos[0]], [nodes_pos[n][1], ch_pos[1]], 'gray', alpha=0.5)

        if chs:
            ch_x = [nodes_pos[ch][0] for ch in chs]
            ch_y = [nodes_pos[ch][1] for ch in chs]
            plt.scatter(ch_x, ch_y, c='red', s=80, marker='*', label='CH')
        plt.scatter(base_station[0], base_station[1], c='green', s=100, marker='^', label='BS')
        plt.title(f'GA Clustering at round {rnd}')
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
