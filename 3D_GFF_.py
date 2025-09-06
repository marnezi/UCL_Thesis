import numpy as np                                  
import matplotlib.pyplot as plt                      
from matplotlib.ticker import FuncFormatter         
from scipy.sparse import diags, kron, eye   

# Lets define the graph settigns
N  = 30  # nodes
mu = 0 # mass less case
seed = 42
rng = np.random.default_rng(seed)

# 1D Dirichlet Laplacian (tridiagonal -1, 2, -1)
T = diags([-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)], offsets=[-1, 0, 1])

L = kron(eye(N), T) + kron(T, eye(N))
A = L.toarray()

# Sample h ~ N(0, A^{-1}) via Cholesky 
Lchol = np.linalg.cholesky(A)         
z = rng.standard_normal(N*N)           
h_sample = np.linalg.solve(Lchol.T, z) 
H_interior = h_sample.reshape(N, N)

# set zero at the boundary
H = np.pad(H_interior, 1, mode='constant', constant_values=0)

#Visualization
x = y = np.linspace(0, 1, N + 2)                    
X, Y = np.meshgrid(x, y, indexing='ij')            

fig = plt.figure(figsize=(14, 6))                   
ax = fig.add_subplot(111, projection='3d')           
ax.set_box_aspect([1.9, 1.5, 1.0])                  

surf = ax.plot_surface(X, Y, H, cmap='Spectral', edgecolor='k', linewidth=0.25, antialiased=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("")

ax.zaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.2f}"))

# formating the line
line_x = np.linspace(0, 1, len(x))
line_y = np.linspace(0, 1, len(y))
line_z = [ H[i, i] for i in range(len(line_x))]
ax.plot(line_x, line_y, line_z, color='black', linewidth=0.5, linestyle='-', alpha=0.9)

plt.tight_layout()
plt.savefig("gff_surface.png", dpi=300, bbox_inches='tight')
plt.show()



                               