import numpy as np
import matplotlib.pyplot as plt

# ----- Custom Data -----
# Example dictionary: {method_name: total_PDR_value}
total_pdr = {
    'LEACH_BASELINE': 99.51,
    'DEEC_BASELINE': 99.63,
    'FUZZY_C_MEANS_BASELINE': 99.71,
    'PSO_BASELINE': 99.73,
    'ACO_BASELINE': 99.76,
    'RLHC_PROPOSED': 99.88,
}

# Optional custom colors (you can add more or remove)
colors = ['blue', 'red', 'green', 'gold', 'orange', 'black']

# ----- Plot: Total PDR (%) vs Method (Bar Chart) -----
plt.figure(figsize=(8, 5))
plt.bar(total_pdr.keys(), total_pdr.values(), color=colors[:len(total_pdr)], edgecolor='black')

# Add labels, title, and grid
plt.ylabel("Total PDR (%)", fontsize=12)
plt.xlabel("Methods", fontsize=12)
plt.title("Total Packet Delivery Ratio Comparison", fontsize=14)
plt.grid(axis='y', linestyle='--')

# Optionally annotate each bar with its value
for i, (method, value) in enumerate(total_pdr.items()):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha='center', fontsize=11)

# Display chart
plt.tight_layout()
plt.show()
