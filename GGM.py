#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:21:58 2025

@author: maria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 16:30:19 2025

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 19:16:26 2025

@author: Maria
"""

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

# Case 1: we know the Covariance matrix and we are trying to see how well our estimation holds 
#Create a random graph using networkx

# Step 1: Create a Erdos Reyni graph to test 
nodes_number = 20 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.25 # probability of # of edges per node (we keep it high so we have less disconnected samples)

np.log(20)/20

er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = 1042)
nx.is_connected(er_igraph)#check connectivity 

pos = nx.spring_layout(er_igraph, seed=seed)

nx.draw(er_igraph, pos=pos, with_labels=True, edge_color='gray', node_size=800, font_size=12)
plt.title("Erdős–Rényi Graph")
plt.show()

#Estimate the Adjacency matrix and Degree matrix 

# Get adjacency matrix
adjacent_matrix = nx.to_numpy_array(er_igraph)
degree_matrix = np.diag(np.sum(adjacent_matrix, axis=1))


# Estimate the Laplacian 
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

#laplacian 
print(laplacian_matrix)

#Configuration settigns

#Check 

number_samples = 7700000
((nodes_number**4)*np.log(nodes_number))/(transition_prob**2)
number_samples
number_samples> (nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))


mass_parameter = 1# we keep this fixed if it is too smaller than one the eigenvalues of sigma will be close to zero and make it ill-conditioned 
etta_parameter = 0.5 # smoothing parameter 

fixed_node = 0 # at least one node equal to zero Dirichlet boundary condition 

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



#------------------------------- run a cross validation method while keeping the constraints to find the most optimal etta value ----------------------------------------------


#check that all the eigenvalues are positive definite 
#check the spectral norm error of the graph laplacian 
# evaluate at the precision matrix 

eta_parameters = np.arange(0.1,20,1)
heat_map_matrix = pd.DataFrame({'Eta_param': eta_parameters,'Spectral error':0,'Forbenius error':0, 'TheoriticaL_q':0})

eta= 4.1
seed = 1042 
ss = np.random.SeedSequence(seed)  
children = ss.spawn(len(eta_parameters))



#Compute the precision matrix
precision_matrix = laplacian_matrix + mass_parameter*np.eye(nodes_number) #precision matrix
ovariance_matrix = np.linalg.inv(precision_matrix)

ovariance_matrix[1,:]
precision_matrix[1,:]


    
for k, eta in enumerate(eta_parameters):
    print(eta)
    rng = np.random.default_rng(children[k])

    etta_parameter = eta # smoothing parameter 



    covariance_matrix_Y = etta_parameter*np.eye(nodes_number)
     
    #generate mutliple samples from Sigma
    x_samples = rng.multivariate_normal(mean = np.zeros(nodes_number),cov= covariance_matrix, size = number_samples)

    y_samples = rng.multivariate_normal(mean = np.zeros(nodes_number), cov= covariance_matrix_Y,size = number_samples)
 

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
            
    # Check the threoritical quarantees
    
    
    L_eta_hat = laplacian_etta_hat
    c_eta = np.sqrt(np.linalg.det(L_eta_hat /etta_parameter))
    L_norm2 = np.linalg.norm(L_eta_hat, 2)
    c_star = 0.5 * c_eta * np.exp(-0.5 * L_norm2**2)    
    log_d = np.log(nodes_number)
    
    
     #  check the concentration bounds
  #  bound = (1 / c_star) * np.sqrt(log_d /number_samples)
  #  error = np.linalg.norm(laplacian_etta_hat - laplacian_etta, ord='fro')/nodes_number
  #  test1 = error<bound
    

    #Check theoreum 3.6
    concentration_error_laplacian= np.linalg.norm(laplacian_etta_hat - laplacian_etta, ord=2)
    test2 = ((max_eigenvalue+mass_parameter+etta_parameter)/(etta_parameter**2))*concentration_error_laplacian<1
    
    #check eigenvalues
    eigvals = np.linalg.eigvalsh(laplacian_etta_hat)
    test3 = np.min(eigvals)>0
        
    precision_matrix_hat = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta_hat)) - etta_parameter*np.eye(nodes_number)
    
    #Lets estimate the forbenius norm for the laplacian and for the precision 
         
    forebenius_norm_error = np.linalg.norm( (precision_matrix_hat-precision_matrix) ,ord = "fro") #overall error
    spectral_norm_error = np.linalg.norm((precision_matrix_hat-precision_matrix), ord=2) #square root of the maximum eigenvalue
    

    #Compare with the theoritical bound - check the number of n 

    empirical_bound = forebenius_norm_error/nodes_number
    theoritical_bound = 1*np.sqrt((np.log(nodes_number)*nodes_number**4)/(number_samples*(transition_prob**4)))
    test1 = empirical_bound<=theoritical_bound
    
    heat_map_matrix.loc[heat_map_matrix['Eta_param'] == eta, 'Forbenius error'] = forebenius_norm_error
    heat_map_matrix.loc[heat_map_matrix['Eta_param'] == eta, 'Spectral error'] = concentration_error_laplacian
    
    
    mask = np.isclose(heat_map_matrix['Eta_param'], eta) 
    heat_map_matrix.loc[mask, 'Forbenius error'] = forebenius_norm_error 
    heat_map_matrix.loc[mask, 'Spectral error'] = spectral_norm_error
    
    if test1 and test2 and test3:
        heat_map_matrix.loc[mask, 'TheoriticaL_q'] = 1


heat_map_matrix.to_csv('eta_parameter_n20.csv', index=False)
         
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
ax.set_ylabel('Forbenius error')
ax.set_title(r'Presicion concentration error against $\eta$ parameter')


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
    0.98, 0.95, r'$n = 10$',
    transform=ax.transAxes,
    ha='right', va='top',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)

plt.show()

