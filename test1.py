import numpy as np
import matplotlib.pyplot as plt

# ----- Custom Data -----
# Example dictionary: {method_name: total_PDR_value}
total_pdr = {
    'LEACH_BASELINE': 1110,
    'DEEC_BASELINE': 4008,
    'FUZZY_C_MEANS': 3862,
    'PSO_BASELINE': 5904,
    'ACO_BASELINE': 3731,
    'RLHC_PROPOSED': 7128,
}

# Optional custom colors (you can add more or remove)
colors = ['blue', 'red', 'green', 'gold', 'orange', 'black']

# ----- Plot: Total PDR (%) vs Method (Bar Chart) -----
plt.figure(figsize=(8, 5))
plt.barh(total_pdr.keys(), total_pdr.values(), color=colors[:len(total_pdr)], edgecolor='black')

# Add labels, title, and grid
plt.xlabel("Numbers of Rounds", fontsize=12)
plt.ylabel("Methods", fontsize=12)
plt.title("Operational Lifetime of the Network", fontsize=14)
plt.grid(axis='y', linestyle='--')

# Optionally annotate each bar with its value
for i, (method, value) in enumerate(total_pdr.items()):
    plt.text(i, value + 0.5, f"{value}", ha='center', fontsize=10)

# Display chart
plt.tight_layout()
plt.show()
