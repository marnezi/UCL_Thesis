import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx

seed = 7 # set seed for reproducibility

# Step 1: Create a Erdos Reyni graph to test 
nodes_number = 8 
transition_prob = 0.4

#check the prob. number
transition_prob > np.log(nodes_number)/nodes_number

#construct the Erdos-Reyni simple with 8 nodes
er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = seed)
nx.is_connected(er_igraph)#check connectivity 

pos = nx.spring_layout(er_igraph, seed=seed)

nx.draw(er_igraph, pos=pos, with_labels=True, node_color = 'orange',width = 3,
            edge_color='black', node_size=1000, font_size=16, edgecolors = 'black')
plt.show()
    

