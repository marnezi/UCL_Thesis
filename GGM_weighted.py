import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import multivariate_normal
import seaborn as sea
import math
from matplotlib import cm
from tqdm import tqdm
import itertools

seed = 1042 # set seed for reproducibility
np.random.seed(seed) 

# Case 1: we know the Covariance matrix and we are trying to see how well our estimation holds 
#Create a random graph using networkx

# Step 1: Create a Erdos Reyni graph to test 
nodes_number = 20 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.25 # probability of # of edges per node (we keep it high so we have less disconnected samples)

np.log(nodes_number)/nodes_number

er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = 1042)
nx.is_connected(er_igraph)#check connectivity 

pos = nx.spring_layout(er_igraph, seed=seed)

nx.draw(er_igraph, pos=pos, with_labels=True, edge_color='gray',node_color="orange", node_size=800, font_size=12)
plt.title("Erdős–Rényi Graph")
plt.show()

#  Add random weights to edges
for u, v in er_igraph.edges():
    er_igraph[u][v]['weight'] = np.random.uniform(0, 10) 

# Step 3: Get weighted adjacency matrix
adjacent_matrix = nx.to_numpy_array(er_igraph, weight='weight')

# Step 4: Build weighted Laplacian
degree_matrix = np.diag(np.sum(adjacent_matrix, axis=1))
laplacian_matrix = degree_matrix - adjacent_matrix
laplacian_matrix

laplacian_matrix = nx.laplacian_matrix(er_igraph).toarray()
max_eigenvalue = np.linalg.eigvals(laplacian_matrix).max()

#Plot adjacency and degree matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Adjacency matrix
sea.heatmap(adjacent_matrix, annot=True, cmap="Blues", cbar=False, square=True, ax=axs[0])
axs[0].set_title("Adjacency Matrix")
axs[0].set_xlabel("Node")
axs[0].set_ylabel("Node")

# Degree matrix
sea.heatmap(degree_matrix, annot=True, cmap="Greens", cbar=False, square=True, ax=axs[1])
axs[1].set_title("Degree Matrix")
axs[1].set_xlabel("Node")
axs[1].set_ylabel("Node")

plt.tight_layout()
plt.savefig("adjacency_degree_side_by_side_nocbar.png", dpi=300, bbox_inches='tight')
plt.show()

#Configuration settigns

#Check 

number_samples = 2000000
((nodes_number**4)*np.log(nodes_number))/(transition_prob**2)
number_samples
number_samples> (nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))

mass_parameter = 1# we keep this fixed if it is too smaller than one the eigenvalues of sigma will be close to zero and make it ill-conditioned 
etta_parameter = 0.5 # smoothing parameter 

# Step 1: Simulate a Gaussian Free Field of a graph many times (independent graphs) and 
# estimate the true Sigma
# now following the methodology estimate the estimate of Sigma -1 (hat)
# check how close they are 
# look at the bounds and which variables have the most impact on the consentration 
# check how good is the recovering procedure 
# what is the error estimate - what influence it the most - plots 


#Compute the precision matrix
precision_matrix = laplacian_matrix + mass_parameter*np.eye(nodes_number) #precision matrix
covariance_matrix = np.linalg.inv(precision_matrix)

covariance_matrix[1,:]
precision_matrix[1,:]

covariance_matrix_Y = etta_parameter*np.eye(nodes_number)

# check the conditional number
np.linalg.cond(precision_matrix)
 
#generate mutliple samples from Sigma
x_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number),cov= covariance_matrix, size = number_samples)
x_samples.shape

y_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number), cov= covariance_matrix_Y,size = number_samples)
y_samples.shape
 
def phi_t(t):
    
    phi = np.mean(np.exp(1j * np.sum(y_samples * (x_samples + t), axis=1)))
    return phi

#Create the Laplacian matrix from the fourier analysis 
laplacian_etta = np.linalg.inv(covariance_matrix + (1/etta_parameter)* np.eye(laplacian_matrix.shape[0]))

laplacian_etta_hat = np.zeros((nodes_number,nodes_number))
e = np.eye(nodes_number)

#estimate phi ouside to make the script more fast (only for the diagonal points)

phi_t_values = [phi_t(e[i]) for i in range(nodes_number)]
phi_t_0 = phi_t(np.zeros(nodes_number))
i=j=1

# Estimate the diagonal and off-diagonal entries in the laplacian matrix 
for i in range(nodes_number):
    print(f"Running iteration i = {i}")
    for j in range(nodes_number):
        if i == j:
            laplacian_etta_hat[i,i] = -2*np.log(np.abs(phi_t_values[i]))+2*np.log(np.abs(phi_t_0))
        else:
            term_1 = (e[i] + e[j]) / np.sqrt(2)
            laplacian_etta_hat[j,i] = laplacian_etta_hat[i,j] = -2*np.log(np.abs(phi_t(term_1)))+np.log(np.abs(phi_t_values[i]))+np.log(np.abs(phi_t_values[j]))
            

laplacian_etta_hat
laplacian_etta

# Check the threoritical quarantees

L_eta_hat = laplacian_etta_hat
c_eta = np.sqrt(np.linalg.det(L_eta_hat /etta_parameter))
L_norm2 = np.linalg.norm(L_eta_hat, 2)
c_star = 0.5 * c_eta * np.exp(-0.5 * L_norm2**2)
print("c_*(eta):", c_star)


log_d = np.log(nodes_number)
bound = (1 / c_star) * np.sqrt(log_d /number_samples)
bound

np.linalg.eigvalsh(laplacian_etta_hat)     

 #  check the concentration bounds
error = np.linalg.norm(laplacian_etta_hat - laplacian_etta, ord='fro')/nodes_number
print("Average Frobenius error per entry:", error)

error<bound

error_norm = np.linalg.norm(laplacian_etta_hat - laplacian_etta, ord=2)

#Check theoreum 3.1
((max_eigenvalue+mass_parameter+etta_parameter)/(etta_parameter**2))*error_norm
((max_eigenvalue+mass_parameter+etta_parameter)/(etta_parameter**2))*error_norm<1
      
eigvals = np.linalg.eigvalsh(laplacian_etta_hat)
      
print("Eigenvalues:", eigvals)
print("Min eigenvalue:", np.min(eigvals))
print("Max eigenvalue:", np.max(eigvals))  
print("Condition number:", np.max(eigvals) / np.min(eigvals))
      
      
# estimate the precision matrix from the laplacian etta
eigvals_diff = np.linalg.eigvalsh(etta_parameter * np.eye(nodes_number) - laplacian_etta)
eigvals_diff

if np.any(eigvals_diff <= 1e-8):
    print("Warning: L_hat is nearly singular!")

    
precision_matrix_hat = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta_hat)) - etta_parameter*np.eye(nodes_number)
precision_matrix_hat

#Lets estimate the forbenius norm for the laplacian and for the precision 
     
forebenius_norm_error = np.linalg.norm( (precision_matrix_hat-precision_matrix) ,ord = "fro") #overall error
forebenius_norm_error

spectral_norm_error = np.linalg.norm((precision_matrix_hat-precision_matrix), ord=2) #square root of the maximum eigenvalue
spectral_norm_error

#Compare with the theoritical bound - check the number of n 

empirical_bound = forebenius_norm_error/nodes_number
theoritical_bound = np.sqrt(np.log(nodes_number)/number_samples)

empirical_bound
theoritical_bound


print("||true_precision||_F =", np.linalg.norm(precision_matrix, ord='fro'))
print("||precision_hat||_F =", np.linalg.norm(precision_matrix_hat, ord='fro'))

eigvals = np.linalg.eigvalsh(etta_parameter * np.eye(nodes_number) - laplacian_etta)
print("Min eig of (eta I - L_eta):", np.min(eigvals))




#---------------------------------------------------------------------------------------------reconstruct Laplacian ---------------------------


#recover the laplacian matrix through the precision hat 
#reconstruct the graph using the laplacian 

#Recover the laplacian through the precisio matrix 

laplacian_GFF  = precision_matrix_hat - mass_parameter*np.eye(nodes_number)

def reconstruct_graph_from_laplacian(laplacian_GFF, threshold=1e-3):

    d = laplacian_GFF.shape[0]
    A_est = np.zeros_like(laplacian_GFF)

    for i in range(d):
        for j in range(i + 1, d):
            if laplacian_GFF[i, j] < -threshold:
                A_est[i, j] = A_est[j, i] = 1

    G = nx.from_numpy_array(A_est)
    return G

# Example usage:
G_hat = reconstruct_graph_from_laplacian(laplacian_GFF, threshold=1e-3)

# Draw it
plt.figure(figsize=(6, 6))
nx.draw(G_hat, with_labels=True, node_color="skyblue", edge_color="gray")
plt.title("Reconstructed Erdős–Rényi Graph from Laplacian")
plt.show()


np.round(laplacian_GFF, decimals=0)
laplacian_matrix

# look at the eigenvalues of laplacian - properties of the graph (0s - connectivity)- check eigenvalue 
#Fiedler value - second eigenvalue - tells you how conencted the graph is 
# average degree of the matrix - global property of Laplacian 

eigenvalues_true = np.linalg.eigvalsh(laplacian_matrix)
eigenvalues_sorted_true = np.sort(eigenvalues_true)

# Fiedler value is the second smallest eigenvalue
fiedler_value_true = eigenvalues_sorted_true[1]
fiedler_value_true

# Average degree is the mean of the diagonal (since L_ii = degree of node i)
average_degree = np.mean(np.diag(laplacian_matrix))
average_degree

eigenvalues_GFF = np.linalg.eigvalsh(laplacian_GFF)
eigenvalues_sorted_GFF = np.sort(eigenvalues_GFF)

# Fiedler value is the second smallest eigenvalue
fiedler_value_GFF = eigenvalues_sorted_GFF[1]
fiedler_value_GFF, fiedler_value_true

# Average degree is the mean of the diagonal (since L_ii = degree of node i)
average_degree_GFF = np.mean(np.diag(laplacian_GFF))
average_degree_GFF, average_degree


#Compare with the vanilla estimator 

# Fird we need to estimate the characteristic function
def psi_n(u, X):
    return np.mean(np.exp(1j * X @ u))

# Parameter U
R = np.linalg.norm(covariance_matrix, ord=2) 
U = R**(-0.5)  # improvised choise

# estitmatorr
cov_BMT = np.zeros((nodes_number, nodes_number))
e = np.eye(nodes_number)

# Diagonal entries
for i in range(nodes_number):
    psi_val = psi_n(U * e[i], x_samples)
    cov_BMT[i, i] = -2 / U**2 * np.real(np.log(psi_val))

# Off-diagonal entries
for i in range(nodes_number):
    for j in range(i + 1, nodes_number):
        t = U * (e[i] + e[j]) / np.sqrt(2)
        psi_val = psi_n(t, x_samples)
        cov_BMT[i, j] = -2 / U**2 * np.real(np.log(psi_val)) - 0.5 * (cov_BMT[i, i] + cov_BMT[j, j])
        cov_BMT[j, i] = cov_BMT[i, j] 

# compute the precision
precision_BMT = np.linalg.inv(cov_BMT)


forebenius_norm_error_BMT = np.linalg.norm( (precision_BMT-precision_matrix) ,ord = "fro") #overall error
forebenius_norm_error_BMT

spectral_norm_error_BMT = np.linalg.norm((precision_BMT-precision_matrix), ord=2) #square root of the maximum eigenvalue
spectral_norm_error_BMT


np.linalg.norm( (precision_matrix_hat-precision_matrix) ,ord = "fro")
np.linalg.norm( (precision_BMT-precision_matrix) ,ord = "fro") #overall error

np.linalg.norm( (precision_matrix_hat-precision_matrix) , ord=2) 
np.linalg.norm((precision_BMT-precision_matrix), ord=2) 

#reconstruct laplacian 

laplacian_vanilla = precision_BMT - mass_parameter*np.eye(nodes_number)
average_degree_laplacian_vanilla = np.mean(np.diag(laplacian_vanilla))

eigenvalues_vanilla = np.linalg.eigvalsh(laplacian_vanilla)
eigenvalues_sorted_vanilla = np.sort(eigenvalues_vanilla)

# Fiedler value is the second smallest eigenvalue
fiedler_value_vanilla = eigenvalues_sorted_vanilla[1]
fiedler_value_GFF, fiedler_value_true,fiedler_value_vanilla
average_degree_GFF, average_degree,average_degree_laplacian_vanilla





