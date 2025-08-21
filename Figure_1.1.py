# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 15:10:59 2025

@author: Maria
"""

import matplotlib.pyplot as plt
import networkx as nx

# πarameters 
d = 8          # number of nodes
p = 0.35       # probability of edge creation
seed = 7       # random seed for reproducibility

# graph
erdos_reyni_graph = nx.erdos_renyi_graph(n=d, p=p, seed=seed)

# Layout for visualization
pos = nx.spring_layout(erdos_reyni_graph, seed=seed)

# Draw
plt.figure(figsize=(5, 5))
nx.draw_networkx_edges(erdos_reyni_graph, pos, edge_color="black", width=1.5)
nx.draw_networkx_nodes(erdos_reyni_graph, pos, node_color="orange", edgecolors="black", linewidths=2, node_size=600)
nx.draw_networkx_labels(erdos_reyni_graph, pos, font_color="black", font_size=12)

plt.axis("off")
plt.tight_layout()

out_path = "erdos_renyi_orange_nodes_black_edges.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

out_path
