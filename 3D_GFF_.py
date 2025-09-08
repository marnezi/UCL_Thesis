import numpy as np                                  
import matplotlib.pyplot as plt                      
from matplotlib.ticker import FuncFormatter           

# Lets define the graph settigns
N  = 35 # this is the interior nodes (we should keep in mind that the exterior nodes will be set to zero)
mu = 0 # massless case so Σ^{-1} =  L
seed = 42
rng = np.random.default_rng(seed)

# 1D Dirichlet Laplacian (tridiagonal -1, 2, -1)
#The numbers 2 and –1 in the discrete Laplacian come from how the second derivative is approximated on a grid. 
#Each interior point is compared to the average of its two neighbors: the central value gets a weight of 2, 
#while each neighbor gets –1. This ensures the matrix correctly measures curvature and 
#makes the quadratic form correspond to the sum of squared differences between adjacent points

# https://en.wikipedia.org/wiki/Discrete_Laplace_operator
# https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
# Saad, Yousef. Iterative methods for sparse linear systems. Society for Industrial and Applied Mathematics, 2003.

diag_xx_yy = np.zeros((N, N))
np.fill_diagonal(diag_xx_yy, 2) #diagonal
np.fill_diagonal(diag_xx_yy[1:], -1)  #below 
np.fill_diagonal(diag_xx_yy[:,1:], -1)  #above 
print(diag_xx_yy)

#2D discrete Laplacian
# https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
# The 2D Laplacian is the sum of the 1D Laplacians in x and y directions

laplacian = np.kron(np.eye(N), diag_xx_yy) + np.kron(diag_xx_yy, np.eye(N)) #  now instead of having just the left and right diagonal neighbours we also get up and down - we went from 1D to 2D 
print(laplacian[:10,:10]) 

# precision matrix
sigma_inv = np.array(laplacian, copy=True)

# The laplacian matrix is a positive define and symmetric matrix 
#  we want to sample from h: N(0,Σ) so we need the Σ
sigma = np.linalg.inv(sigma_inv)   #this as we will saw later in the thesis is not a good practice as it can introduce instabilities but here we do it for the same of the plot
      
h_sample = rng.multivariate_normal(mean=np.zeros(N*N), cov=sigma)
points_interior_matrix = h_sample.reshape(N, N)

# set zero at the boundary
surface = np.pad(points_interior_matrix, 1, mode='constant', constant_values=0)

#Visualization
x = y = np.linspace(0, 1, N + 2)                    
X, Y = np.meshgrid(x, y, indexing='ij')            

fig = plt.figure(figsize=(14, 6))                   
ax = fig.add_subplot(111, projection='3d')           
ax.set_box_aspect([1.9, 1.5, 1.0])                  

surf = ax.plot_surface(X, Y, surface, cmap='Spectral', edgecolor='k', linewidth=0.25, antialiased=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("")

ax.zaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.2f}"))

# formating the line
line_x = np.linspace(0, 1, len(x))
line_y = np.linspace(0, 1, len(y))
line_z = [ surface[i, i] for i in range(len(line_x))]
ax.plot(line_x, line_y, line_z, color='black', linewidth=0.5, linestyle='-', alpha=0.9)

plt.tight_layout()
plt.savefig("gff_surface.png", dpi=300, bbox_inches='tight')
plt.show()

                               




                               
