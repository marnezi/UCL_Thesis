import matplotlib.pyplot as plt
import networkx as nx

# parameters 
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


heatmap_data = heat_map_matrix[['Eta_param', 'Forbenius error', 'TheoriticaL_q']].to_numpy()

          
plt.imshow(heatmap_data, aspect='auto')
plt.colorbar(label='Value')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Heatmap of Eta_param, Forbenius, TheoriticaL_q')
plt.show()          
    
heat_map_matrix = heat_map_matrix.sort_values(by='Forbenius error')
heat_map_matrix[heat_map_matrix['TheoriticaL_q'] == 1]


plt.scatter(
    heat_map_matrix[heat_map_matrix['TheoriticaL_q'] == 1]['Eta_param'],
    heat_map_matrix[heat_map_matrix['TheoriticaL_q'] == 1]['Forbenius error'],
    color='black'
)
plt.xlabel('Eta_param')
plt.ylabel('Forbenius')
plt.title('Point Eta parameter versus Forbenius error')
plt.show()



df = heat_map_matrix.sort_values('Eta_param')  # ensure ordered
x = df['Eta_param'].to_numpy()
y = df['Forbenius error'].to_numpy()
q1 = (df['TheoriticaL_q'] == 1).to_numpy()

fig, ax = plt.subplots()
ax.scatter(x, y, color='black')  # black points
ax.set_xlabel(r'$\eta$ parameter')
ax.set_ylabel('Forbenius error $||\hat{\Sigma}^{-1} - \Sigma^{-1}||/d$')
ax.set_title(r'Concentration error against $\eta$ parameter')


dx = np.median(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 0.5
i = 0
while i < len(x):
    if q1[i]:
        j = i
        while j < len(x) and q1[j]:
            j += 1
        ax.axvspan(x[i] - dx/2, x[j-1] + dx/2, color='orange', alpha=0.3)
        ax.axvline(x[i] - dx/2, color='orange', linestyle='--')
        ax.axvline(x[j-1] + dx/2, color='orange', linestyle='--')
        i = j
    else:
        i += 1

# Text box instead of legend dot
ax.text(
    0.98, 0.95, r'$n = 20$',
    transform=ax.transAxes,
    ha='right', va='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)

plt.show()

