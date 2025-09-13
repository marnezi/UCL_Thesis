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

# Case 1: We know the Covariance matrix, and we are trying to see how well our estimation holds 
#Create a random graph using networkx

# Step 1: Create an Erdos-Reyni graph to test 
nodes_number = 10 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.6 # probability of # of edges per node (we keep it high so we have fewer disconnected samples)

np.log(10)/10

er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = 1042)
nx.is_connected(er_igraph)#check connectivity 

pos = nx.spring_layout(er_igraph, seed=seed)

nx.draw(er_igraph, pos=pos, with_labels=True, edge_color='gray', node_size=800, font_size=12)
plt.title("Erdős–Rényi Graph")
plt.show()

#Configuration settings

number_samples = 100000
mass_parameter = 1000# we keep this fixed, if it is too small, then the eigenvalues of sigma will be close to zero and make it ill-conditioned 
etta_parameter = 1100 # smoothing parameter 

fixed_node = 0 # at least one node equal to zero, Dirichlet boundary condition 

#Check 
(nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))
number_samples
number_samples> (nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))


# Step 1: Simulate a Gaussian Free Field of a graph many times (independent graphs) and 
# estimate the true Sigma
# now following the methodology, estimate the estimate of Sigma -1 (hat)
# check how close they are 
# look at the bounds and which variables have the most impact on the concentration 
# check how good the recovery procedure is 
# what is the error estimate - what influences it the most - plots 

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

#estimate phi outside to make the script faster (only for the diagonal points)

phi_t_values = [phi_t(e[i]) for i in range(nodes_number)]
phi_t_0 = phi_t(np.zeros(nodes_number))
i=j=1

# Estimate the diagonal and off-diagonal entries in the Laplacian matrix 
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
      
      
# estimate the precision matrix from the Laplacian etta
eigvals_diff = np.linalg.eigvalsh(etta_parameter * np.eye(nodes_number) - laplacian_etta)
eigvals_diff

if np.any(eigvals_diff <= 1e-8):
    print("⚠️ Warning: eta I - L_hat is nearly singular!")

     
precision_matrix_hat = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta)) - etta_parameter*np.eye(nodes_number)
   
# Let's estimate the Frobenius norm for the Laplacian and for the precision 
     
forebenius_norm_error = np.linalg.norm( (precision_matrix_hat-precision_matrix) ,ord = "fro") #overall error
forebenius_norm_error

spectral_norm_error = np.linalg.norm((precision_matrix_hat-precision_matrix), ord=2) #square root of the maximum eigenvalue
spectral_norm_error

#Compare with the theoretical bound - check the number of n 

empirical_bound = forebenius_norm_error/nodes_number
theoritical_bound = np.sqrt(np.log(nodes_number)/number_samples)

empirical_bound
theoritical_bound


print("||true_precision||_F =", np.linalg.norm(precision_matrix, ord='fro'))
print("||precision_hat||_F =", np.linalg.norm(precision_matrix_hat, ord='fro'))

eigvals = np.linalg.eigvalsh(etta_parameter * np.eye(nodes_number) - laplacian_etta)
print("Min eig of (eta I - L_eta):", np.min(eigvals))


def psi_n(u):
    return np.mean(np.exp(1j * x_samples @ u))

U = np.sqrt(np.log(number_samples))  # we can try different u settings
psi_matrix = np.zeros((nodes_number, nodes_number), dtype=np.float64)

for i in range(nodes_number):
    psi_matrix[i, i] = -2 / U**2 * np.log(np.abs(psi_n(U * e[i])))

for i in range(nodes_number):
    for j in range(i+1, nodes_number):
        term = (e[i] + e[j]) / np.sqrt(2)
        psi_matrix[i, j] = psi_matrix[j, i] = -2 / U**2 * np.log(np.abs(psi_n(U * term))) - 0.5 * (psi_matrix[i, i] + psi_matrix[j, j])

covariance_hat_vanilla = psi_matrix
precision_hat_vanilla = np.linalg.inv(covariance_hat_vanilla)

np.linalg.norm( (precision_hat_vanilla-precision_matrix) ,ord = "fro") #overall error













