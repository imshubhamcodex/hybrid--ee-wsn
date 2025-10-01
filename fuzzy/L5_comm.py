from proposed.utils import energy_tx, energy_rx, distance
from proposed.config import PACKET_SIZE_BITS


def radio_comm(clusters, chs, nodes_energy, nodes_pos, base_station, tx_power_factor, successful_packets=0, total_packets=0):
        # For each cluster, members transmit to CH, CH aggregates and forwards to BS (single hop for demo)
    for cidx, members in enumerate(clusters):
        ch = chs[cidx]
        # each member sends one packet (unless dead)
        for node in members:
            if nodes_energy[node] <= 0: continue
            d = distance(nodes_pos[node], nodes_pos[ch])
            etx = energy_tx(PACKET_SIZE_BITS, d) * tx_power_factor
            erx = energy_rx(PACKET_SIZE_BITS)
            # member transmits (energy deducted)
            nodes_energy[node] -= etx
            # CH receives (if alive)
            if nodes_energy[ch] > 0:
                nodes_energy[ch] -= erx * 0  # often CH rx cost borne by CH; set to zero double counting prevention (simplify)
                successful_packets += 1
            total_packets += 1
            # if node died mid-transmission, treat as lost (we still counted energy used)
            if nodes_energy[node] <= 0:
                # lost
                successful_packets -= 1  # approximate
    # CH -> BS transmission (aggregated single packet per CH)
    for cidx, ch in enumerate(chs):
        if nodes_energy[ch] <= 0: continue
        d = distance(nodes_pos[ch], base_station)
        etx = energy_tx(PACKET_SIZE_BITS, d) * tx_power_factor
        nodes_energy[ch] -= etx
        total_packets += 1
        if nodes_energy[ch] > 0:
            successful_packets += 1

    return nodes_energy, successful_packets, total_packets