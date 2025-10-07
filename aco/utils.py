import math
import numpy as np
from proposed.config import E_ELEC, EPS_FS, EPS_MP, D0, STATE_BINS_E, INIT_ENERGY

def energy_tx(bits, d):
    """Transmission energy for bits over distance d (meters)."""
    if d < D0:
        return bits * E_ELEC + bits * EPS_FS * (d**2)
    else:
        return bits * E_ELEC + bits * EPS_MP * (d**4)

def energy_rx(bits):
    return bits * E_ELEC

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def discretize(value, bins):
    return int(np.digitize([value], bins=bins)[0])


def build_bins():
    # energy bins based on initial energy range [0, INIT_ENERGY]
    e_bins = np.linspace(0, INIT_ENERGY, STATE_BINS_E+1)[1:-1].tolist()
    load_bins = [1.2, 1.5]  # ratio thresholds for cluster imbalance
    pdr_bins = [0.90, 0.97]  # pdr bins
    return e_bins, load_bins, pdr_bins

def state_from_metrics(avg_energy, cluster_sizes, pdr, LOAD_BINS, PDR_BINS):
    # E_avg_bucket (0..STATE_BINS_E-1)
    e_bucket = discretize(avg_energy/INIT_ENERGY, bins=np.linspace(0,1,STATE_BINS_E+1)[1:-1])
    # load bucket: max cluster size / mean cluster size
    if len(cluster_sizes)==0:
        load_ratio = 1.0
    else:
        load_ratio = max(cluster_sizes)/ (np.mean(cluster_sizes)+1e-12)

    load_bucket = discretize(load_ratio, bins=LOAD_BINS)
    pdr_bucket = discretize(pdr, bins=PDR_BINS)
    return (e_bucket, load_bucket, pdr_bucket)

def measure_metrics(nodes_energy, clusters, chs, successful_packets, total_packets):
    alive = np.sum(nodes_energy > 0)
    avg_e = np.mean(nodes_energy)
    pdr = successful_packets / (total_packets+1e-12)
    cluster_sizes = [len(c) for c in clusters] if clusters else []
    return avg_e, pdr, alive, cluster_sizes