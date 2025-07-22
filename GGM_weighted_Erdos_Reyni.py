#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:32:57 2025

@author: maria
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
import optuna


seed = 1042 # set seed for reproducibility

# Case 1: we know the Covariance matrix and we are trying to see how well our estimation holds 
#Create a random graph using networkx

# Step 1: Create a Erdos Reyni graph to test 
nodes_number = 8 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.6 # probability of # of edges per node (we keep it high so we have less disconnected samples)

er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = 1042)
nx.is_connected(er_igraph)#check connectivity 

nx.draw(er_igraph,with_labels=True, edge_color='gray', node_size=800, font_size=12)

# Show the plot
plt.title(f"Erdős–Rényi Graph")
plt.show()


#Configuration settigns

number_samples = 100000
mass_parameter = 1000# we keep this fixed if it is too smaller than one the eigenvalues of sigma will be close to zero and make it ill-conditioned 
etta_parameter = 1100 # smoothing parameter 

fixed_node = 0 # at least one node equal to zero Dirichlet boundary condition 

#Check 
(nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))
number_samples
number_samples> (nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))


# Step 1: Simulate a Gaussian Free Field of a graph many times (independent graphs) and 
# estimate the true Sigma
# now following the methodology estimate the estimate of Sigma -1 (hat)
# check how close they are 
# look at the bounds and which variables have the most impact on the consentration 
# check how good is the recovering procedure 
# what is the error estimate - what influence it the most - plots 

#  Add random weights to edges
for u, v in er_igraph.edges():
    er_igraph[u][v]['weight'] = np.random.uniform(0.5, 2.0)  # weight in [0.5, 2.0]

# Step 3: Get weighted adjacency matrix
adjacent_matrix = nx.to_numpy_array(er_igraph, weight='weight')

# Step 4: Build weighted Laplacian
degree_matrix = np.diag(np.sum(adjacent_matrix, axis=1))
laplacian_matrix = degree_matrix - adjacent_matrix
laplacian_matrix


max_eigenvalue = np.linalg.eigvals(laplacian_matrix).max()

#Compute the precision matrix
precision_matrix = laplacian_matrix + mass_parameter*np.eye(nodes_number) #precision matrix
covariance_matrix = np.linalg.inv(precision_matrix)

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
            

laplacian_etta_hat.shape
np.linalg.eigvalsh(laplacian_etta_hat)     

 #  check the concentration bounds
error_norm = np.linalg.norm((laplacian_etta_hat-laplacian_etta), ord=2)
((max_eigenvalue+mass_parameter+etta_parameter)/(etta_parameter**2))*error_norm
      
eigvals = np.linalg.eigvalsh(laplacian_etta_hat)
      
print("Eigenvalues:", eigvals)
print("Min eigenvalue:", np.min(eigvals))
print("Max eigenvalue:", np.max(eigvals))  
print("Condition number:", np.max(eigvals) / np.min(eigvals))
      
      
# estimate the precision matrix from the laplacian etta
eigvals_diff = np.linalg.eigvalsh(etta_parameter * np.eye(nodes_number) - laplacian_etta)
eigvals_diff

if np.any(eigvals_diff <= 1e-8):
    print("⚠️ Warning: eta I - L_hat is nearly singular!")

     
precision_matrix_hat = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta)) - etta_parameter*np.eye(nodes_number)
   
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




pos = nx.spring_layout(er_igraph)
edges = nx.get_edge_attributes(er_igraph, 'weight')
nx.draw(er_igraph, pos, with_labels=True, node_color='lightblue')
nx.draw_networkx_edge_labels(er_igraph, transition_prob, edge_labels={e: f"{w:.2f}" for e, w in edges.items()})

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example: assume you already have the matrices
# true_L_eta: the true Laplacian with smoothing
# est_L_eta: your estimated Laplacian from Fourier transform
# If needed, project to PSD before plotting

# Compute difference matrix
diff = laplacian_etta_hat - laplacian_etta

# Set up the 3-panel plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(laplacian_etta, ax=axes[0], cmap='viridis', square=True, cbar=True)
axes[0].set_title('True $L(\\eta)$')

sns.heatmap(laplacian_etta_hat, ax=axes[1], cmap='viridis', square=True, cbar=True)
axes[1].set_title('Estimated $\\hat{L}(\\eta)$')

sns.heatmap(diff, ax=axes[2], cmap='coolwarm', center=0, square=True, cbar=True)
axes[2].set_title('Difference $\\hat{L}(\\eta) - L(\\eta)$')

plt.tight_layout()
plt.show()


















