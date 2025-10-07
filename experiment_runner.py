import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from proposed.RL_PROPOSED import run_proposed
from proposed.config import INIT_ENERGY, N_NODES
from leach.LEACH_BASELINE import run_leach
from deec.DEEC_BASELINE import run_deec
from fuzzy.FUZZY_C_MEANS_BASELINE import run_fuzzy
from pso.PSO_BASELINE import run_pso
from aco.ACO_BASELINE import run_aco

METHODS = {
    'LEACH_BASELINE': run_leach,
    'DEEC_BASELINE': run_deec,
    'FUZZY_C_MEANS_BASELINE': run_fuzzy,
    'PSO_BASELINE': run_pso,
    'ACO_BASELINE': run_aco,
    'RL_PROPOSED': run_proposed,
}

results = {}

if __name__ == "__main__":
    # Run all methods
    for method_name, method_func in METHODS.items():
        print(f"Running method: {method_name}")
        (
            avg_energy_history, pdr_history, alive_nodes_history, num_ch_history,
            reward_history, epsilon_history, throughput_history, pdr_percent_history,
            first_dead_round, half_dead_round, last_dead_round
        ) = method_func()
        
        results[method_name] = {
            "avg_energy_history": avg_energy_history,
            "pdr_history": pdr_history,
            "alive_nodes_history": alive_nodes_history,
            "num_ch_history": num_ch_history,
            "reward_history": reward_history,
            "epsilon_history": epsilon_history,
            "throughput_history": throughput_history,
            "pdr_percent_history": pdr_percent_history,
            "first_dead_round": first_dead_round,
            "half_dead_round": half_dead_round,
            "last_dead_round": 1000
        }

    # ----- Plot 1: Dead Nodes vs Number of Rounds -----
    plt.figure(figsize=(10,6))
    for method_name, res in results.items():
        rounds = range(1, res["last_dead_round"] + 1)
        dead_nodes = [N_NODES - a for a in res["alive_nodes_history"][:res["last_dead_round"]]]
        plt.plot(rounds, dead_nodes, label=method_name, linestyle="--")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Dead Nodes")
    plt.title("Dead Nodes vs Number of Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- Plot 2: Alive Nodes vs Number of Rounds -----
    plt.figure(figsize=(10,6))
    for method_name, res in results.items():
        rounds = range(1, res["last_dead_round"] + 1)
        alive_nodes = res["alive_nodes_history"][:res["last_dead_round"]]
        plt.plot(rounds, alive_nodes, label=method_name, linestyle="--")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Alive Nodes")
    plt.title("Alive Nodes vs Number of Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- Plot 3: Cumulative Throughput vs Number of Rounds -----
    plt.figure(figsize=(10,6))
    for method_name, res in results.items():
        rounds = range(1, res["last_dead_round"] + 1)
        cumulative_throughput = np.cumsum(res["throughput_history"][:res["last_dead_round"]])
        plt.plot(rounds, cumulative_throughput, label=method_name, linestyle="--")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Cumulative Throughput")
    plt.title("Cumulative Throughput vs Number of Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- Plot 4: Total Energy Consumed vs Number of Rounds -----
    plt.figure(figsize=(10,6))
    for method_name, res in results.items():
        rounds = range(1, res["last_dead_round"] + 1)
        total_energy_consumed = [INIT_ENERGY - e for e in res["avg_energy_history"][:res["last_dead_round"]]]
        total_energy_consumed_all_nodes = np.array(total_energy_consumed) * N_NODES
        plt.plot(rounds, total_energy_consumed_all_nodes, label=method_name, linestyle="--")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Total Energy Consumed")
    plt.title("Total Energy Consumption vs Number of Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----- Plot 5: Total PDR (%) vs Method (Bar Chart) -----
    total_pdr = {method_name: np.mean(res["pdr_percent_history"][:res["last_dead_round"]]) 
                 for method_name, res in results.items()}
    plt.figure(figsize=(8,5))
    plt.bar(total_pdr.keys(), total_pdr.values(), color='skyblue')
    plt.ylabel("Total PDR (%)")
    plt.title("Total Packet Delivery Ratio Comparison")
    plt.grid(True)
    plt.show()

    # ------ Plot 6: Chart Horizontal for All Node Dead for each Algo -----------



    # ------ Table: First, Half, Last Dead Node Rounds -----
    death_table = pd.DataFrame({
        "Method": list(results.keys()),
        "First Node Death": [res["first_dead_round"] for res in results.values()],
        "Half Nodes Death": [res["half_dead_round"] for res in results.values()],
        "Last Node Death": [res["last_dead_round"] for res in results.values()]
    })
    print("\nNode Death Summary Table:")
    print(death_table)
