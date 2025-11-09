# 📡 RL-Driven Hybrid Clustering for Energy-Efficient Wireless Sensor Networks (EE-WSN)

## 🧠 Overview
This project presents a **Reinforcement Learning (RL)-driven Hybrid Clustering Framework** for **Energy-Efficient Wireless Sensor Networks (EE-WSNs)**. The system integrates **hierarchical clustering**, **optimization-based CH selection**, and **Q-learning–based adaptive tuning** to improve energy efficiency, packet delivery ratio (PDR), and network lifetime.

The proposed model fuses multiple intelligent layers:
- **L1:** DEEC-based candidate selection  
- **L2:** EEKA-based optimal CH refinement  
- **L3:** K-Means clustering for compact cluster formation  
- **L4:** Q-learning for dynamic re-clustering and transmission tuning  
- **L5:** Energy-aware communication model  

This hybrid approach ensures balanced energy consumption, adaptive clustering, and improved data throughput over traditional WSN protocols.

---

## ⚙️ Key Features
- **Multi-layer hybrid design** combining DEEC, EEKA, and K-Means.
- **Q-learning-driven local decision-making** for CH re-selection and power control.
- **Energy-aware radio communication model** simulating realistic energy depletion.
- **Dynamic epsilon decay** for exploration–exploitation trade-off.
- **Statistical logging** for energy, PDR, rewards, throughput, and lifetime metrics.
- **Visualization support** for cluster formation at periodic rounds.

---

## 🧩 Project Structure
```
RL_Hybrid_EEWSN/
│
├── proposed/
│   ├── config.py          # Simulation parameters and constants
│   ├── L1_deec.py         # DEEC-based CH candidate selection
│   ├── L2_eeka.py         # EEKA-based optimal CH refinement
│   ├── L3_kmean.py        # K-Means clustering
│   ├── L4_rlql.py         # Q-learning: choose_action, compute_reward, update_q
│   ├── L5_comm.py         # Radio communication energy model
│   ├── utils.py           # Helper utilities (distance, metrics, binning)
│
├── run_proposed.py        # Main simulation script (entry point)
└── README.md              # Project documentation
```

---

## 🧮 Methodology Summary

| Layer | Function | Description |
|:------|:----------|:-------------|
| **L1** | `deec_select()` | Selects candidate CHs based on DEEC probability proportional to residual energy. |
| **L2** | `ee_ka_select()` | Refines CHs using Energy-Efficient K-Means Approximation (EEKA) to minimize intra-cluster distance. |
| **L3** | `kmeans()` | Forms compact clusters around initial centroids obtained from L2. |
| **L4** | `choose_action()`, `update_q()` | Q-learning module optimizes CH reassignments, power scaling, and load balancing. |
| **L5** | `radio_comm()` | Simulates data transmission and reception with realistic radio energy consumption. |

---

## 📊 Metrics Tracked
During simulation, the following performance metrics are logged each round:
- **Average Energy Consumption**
- **Packet Delivery Ratio (PDR)**
- **Alive Node Count**
- **Number of CHs**
- **Throughput (Successful Packets)**
- **Reward and Epsilon (Q-learning Progress)**
- **First / Half / Last Node Death Rounds**

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install numpy matplotlib
```

### 2️⃣ Run the Simulation
Execute the main script:
```bash
python run_proposed.py
```

### 3️⃣ (Optional) Enable Cluster Plotting
Inside `run_proposed.py`, uncomment the line:
```python
# plot_cluster(rnd, clusters, nodes_pos, chs, base_station)
```
This will visualize clustering every 1000 rounds.

---

## 🧠 Q-Learning Agent Description
| Component | Purpose |
|:-----------|:----------|
| **State Representation** | Derived from normalized energy levels, cluster load distribution, and PDR bins. |
| **Action Space** | {`NOOP`, `REASSIGN_FEW`, `SWITCH_CH`, `REDUCE_TX`, `INCREASE_TX`} |
| **Reward Function** | Balances energy conservation, successful transmission, and action cost. |
| **Policy** | ε-greedy with dynamic decay. |
| **Q-Update Rule** | Standard temporal-difference update based on measured metrics. |

---

## 📈 Sample Outputs
- **Network Lifetime:** rounds till all nodes die  
- **FDR / HDR / LDR:** first, half, and last node death rounds  
- **Plots:** Energy decay, alive nodes, reward trends, and cluster snapshots  

---

## 🧪 Example Results (Illustrative)
| Metric | Description | Value (Example) |
|:--------|:-------------|:----------------|
| First Node Dead | Network stability period | ~900 rounds |
| Half Nodes Dead | Mid lifetime | ~3100 rounds |
| Last Node Dead | Network lifetime | ~5400 rounds |
| Avg. PDR | Packet delivery ratio | >94% |
| Energy Balance | Uniform | Yes |

---

## 🧰 Customization
Modify parameters in `proposed/config.py`:
```python
N_NODES = 100
AREA_SIZE = 100
INIT_ENERGY = 2.0
ROUNDS = 6000
EPSILON = 0.8
NUM_CLUSTERS = 5
```

---

## 🧾 Citation (if used in academic work)
If you use or modify this implementation, please cite:
> **"RL-driven Hybrid Clustering for Energy-Efficient Wireless Sensor Networks"**,  
> Shubham Kumar, 2025.

---

## 📬 Contact
**Author:** Shubham Kumar  
**Email:** [your.email@example.com]  
**Field:** Wireless Sensor Networks | Reinforcement Learning | IoT Optimization  
