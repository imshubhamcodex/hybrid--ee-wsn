import matplotlib.pyplot as plt

def plot_metrics(avg_energy_history, pdr_history, alive_nodes_history, num_ch_history, reward_history, epsilon_history, throughput_history, pdr_percent_history):
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(avg_energy_history, label='Avg Energy')
    plt.xlabel('Rounds')
    plt.ylabel('Energy')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(pdr_history, label='Packet Delivery Ratio')
    plt.xlabel('Rounds')
    plt.ylabel('PDR')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(alive_nodes_history, label='Alive Nodes')
    plt.xlabel('Rounds')
    plt.ylabel('Alive Nodes')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(num_ch_history, label='Number of CHs')
    plt.xlabel('Rounds')
    plt.ylabel('CHs')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(reward_history, label='Reward')
    plt.xlabel('Rounds')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(epsilon_history, label='Epsilon')
    plt.xlabel('Rounds')
    plt.ylabel('Epsilon')
    plt.legend()

    plt.tight_layout()
    plt.show()