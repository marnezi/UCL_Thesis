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
import optuna


seed = 1042 # set seed for reproducibility

# Case 1: we know the Covariance matrix and we are trying to see how well our estimation holds 
#Create a random graph using networkx

#Configuration settigns
nodes_number = 10 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.6 # probability of # of edges per node (we keep it high so we have less disconnected samples)
number_samples = 100000
mass_parameter = 100 # we keep this fixed
etta_parameter = 19 # smoothing parameter 

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


# Step 1: Create a Erdos Reyni graph to test 

er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob,seed = 1042)
nx.is_connected(er_igraph)#check connectivity 

nx.draw(er_igraph,with_labels=True, edge_color='gray', node_size=800, font_size=12)

# Show the plot
plt.title(f"Erdős–Rényi Graph")
plt.show()

# Estimate the Laplacian 
adjacent_matrix = nx.to_numpy_array(er_igraph)
degree_matrix = np.diag(np.sum(adjacent_matrix, axis=1))
laplacian_matrix = degree_matrix - adjacent_matrix
laplacian_matrix

#Compute Sigma -1
sigma_inv_true = laplacian_matrix + mass_parameter*np.eye(nodes_number) #precision matrix
sigma_true = np.linalg.inv(sigma_inv_true)
 
#generate mutliple samples from Sigma
x_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number), cov= sigma_true, size = number_samples)
y_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number), cov= etta_parameter*np.eye(nodes_number),size = number_samples)
 
def phi_t(t):
    phi = np.mean(np.exp(1j * np.sum(y_samples * (x_samples + t), axis=1)))
    return phi

#Create the Laplacian matrix from the fourier analysis 

laplacian_etta = np.zeros((nodes_number,nodes_number))
e = np.eye(nodes_number)

#estimate phi ouside to make the script more fast (only for the diagonal points)

phi_t_values = [phi_t(e[i]) for i in range(nodes_number)]
phi_t_0 = phi_t(np.zeros(nodes_number))
i=j=1

# Estimate the diagonal and off-diagonal entries in the laplacian matrix 
for i in range(nodes_number):
    for j in range(nodes_number):
        if i == j:
            laplacian_etta[i,i] = -2*np.log(np.abs(phi_t_values[i]))+2*np.log(np.abs(phi_t_0))
        else:
            term_1 = (e[i] + e[j]) / np.sqrt(2)
            laplacian_etta[j,i] = laplacian_etta[i,j] = -2*np.log(np.abs(phi_t(term_1)))+np.log(np.abs(phi_t_values[i]))+np.log(np.abs(phi_t_values[j]))
            
sigma_inv = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta)) - etta_parameter*np.eye(nodes_number)

#Compare with the theoritical bound 
# Estimate the concentration bounds 

laplacian_etta_p = np.linalg.inv(sigma_true + np.eye(nodes_number) / etta_parameter)  # Ground truth L^(η)

forebenius_norm_error = np.linalg.norm( (laplacian_etta-laplacian_etta_p) ,ord = "fro") #overall error
spectral_norm_error = np.linalg.norm((laplacian_etta-laplacian_etta_p), ord=2) #square root of the maximum eigenvalue

max_eigen_value = np.max(np.linalg.eigvalsh(laplacian_matrix))
term1 = (max_eigen_value+mass_parameter+etta_parameter)/(etta_parameter**2)
term2 = (etta_parameter**4)*(1 - term1*spectral_norm_error)
term3 = (max_eigen_value+mass_parameter+etta_parameter)**2

term4 = term3/(term2*nodes_number)

theoretical_bound = term4*forebenius_norm_error
theoretical_bound

empirical_error = np.linalg.norm( (sigma_inv-sigma_inv_true) ,ord = "fro")/nodes_number
empirical_error

assert (term1 * spectral_norm_error) < 1, "Theorem 3.1 condition not satisfied!"
#This condition ensures that the matrix inverse exists and that the Taylor approximation (used to derive the bound) is valid. 
#It’s like a "safe region" for linearization.

#Currently our model does not satisfy the conditions - Bayesian optimization (or random search) to find the set of parameters best for our model 
# we will try multi objective hyperparameter tunning 

C = 5
delta = 0.5


def forbenius_error(nodes_number, mass_parameter, etta_parameter, number_samples, seed=142):
    
    while True:
        er_igraph = nx.erdos_renyi_graph(n=nodes_number, p=transition_prob)
        if nx.is_connected(er_igraph):#check connectivity 
            break
        
    # Estimate the Laplacian 
    adjacent_matrix = nx.to_numpy_array(er_igraph)
    degree_matrix = np.diag(np.sum(adjacent_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacent_matrix
    
    #Compute Sigma -1
    sigma_inv_true = laplacian_matrix + mass_parameter*np.eye(nodes_number) #precision matrix
    sigma_true = np.linalg.inv(sigma_inv_true)
    
    #generate mutliple samples from Sigma
    x_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number), cov= sigma_true, size = number_samples)
    y_samples = np.random.multivariate_normal(mean = np.zeros(nodes_number), cov= etta_parameter*np.eye(nodes_number),size = number_samples)
    
    def phi_t(t):
        phi = np.mean(np.exp(1j * np.sum(y_samples * (x_samples + t), axis=1)))
        return phi
    
    #Create the Laplacian matrix from the fourier analysis 

    laplacian_etta = np.zeros((nodes_number,nodes_number))
    e = np.eye(nodes_number)

    #estimate phi ouside to make the script more fast (only for the diagonal points)

    phi_t_values = [phi_t(e[i]) for i in range(nodes_number)]
    phi_t_0 = phi_t(np.zeros(nodes_number))
    i=j=1

    # Estimate the diagonal and off-diagonal entries in the laplacian matrix 
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i == j:
                laplacian_etta[i,i] = -2*np.log(np.abs(phi_t_values[i]))+2*np.log(np.abs(phi_t_0))
            else:
                term_1 = (e[i] + e[j]) / np.sqrt(2)
                laplacian_etta[j,i] = laplacian_etta[i,j] = -2*np.log(np.abs(phi_t(term_1)))+np.log(np.abs(phi_t_values[i]))+np.log(np.abs(phi_t_values[j]))
                
    
    #check theorem 3.1: spectral norm condition for invertibility
    
    laplacian_etta_p = np.linalg.inv(sigma_true + np.eye(nodes_number) / etta_parameter)  # Ground truth L^(η)
    
    max_eigen = np.max(np.linalg.eigvalsh(laplacian_matrix))
    spectral_norm_error = np.linalg.norm((laplacian_matrix-laplacian_etta_p), ord=2) #square root of the maximum eigenvalue
    term = (max_eigen + mass_parameter + etta_parameter) / etta_parameter**2 * spectral_norm_error

    if term >= 1:
        return 1e6  # Violates spectral condition — penalize   
    
    #c_etta = np.sqrt(np.linalg.det((1/etta_parameter)*laplacian_etta_p))
    #c_star_etta = 0.5*c_etta*np.exp(-0.5*(np.linalg.norm(laplacian_etta_p)**2))
    #delta = 0.05
    #C1 = 10
 #   C2 = 10
    
    #theoritical_bound = np.log(nodes_number**2 *C1/delta)/(C2*(c_star_etta**2))
    
    #lets check constrains 
    #if number_samples < (C*(etta_parameter**2)*(np.log(nodes_number) - np.log(delta))):
    #    return 1e6  # Violates concentration condition - return very large error 
    #
    sigma_inv = etta_parameter**2*(np.linalg.inv(etta_parameter*np.eye(nodes_number) - laplacian_etta)) - etta_parameter*np.eye(nodes_number)
    error = np.linalg.norm(sigma_inv - sigma_inv_true, ord='fro')/nodes_number
    return error


# check the parameters 

def objective(trial):
    etta_parameter = trial.suggest_float("etta_parameter", 10, 1000, log=True)
    mass_parameter = trial.suggest_float("mass_parameter", 0.01, 0.3)
    number_samples = trial.suggest_int("number_samples", 1000, 100000)

    error = forbenius_error(nodes_number,mass_parameter,etta_parameter,number_samples)
    trial.set_user_attr("feasible", error < 1e6)  # label feasible trials
    return error

# === Create and Run Study ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

# === Print Best Results ===
print("✅ Best Parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("📉 Best Frobenius Error:", study.best_value)

df_trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
df_trials["feasible"] = df_trials["user_attrs_feasible"]

# === Plot: Feasible vs Infeasible ===
sea.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sea.scatterplot(data=df_trials, x="params_mass_parameter", y="value", hue="feasible", palette={True: "green", False: "red"})
plt.yscale("log")
plt.xlabel("eta")
plt.ylabel("Frobenius Error")
plt.title("Feasible vs. Infeasible Trials (by Theorem Constraints)")
plt.legend(title="Feasible?")
plt.show()


df_trials.columns

#Create some usefull graphs to see how the error flactuates based on etta, mu parameter and the number of samples 

mass_parameter_values = [0.01, 0.1, 0.2,1]
etta_parameter_values = [0.1, 0.5, 1.0,3]
number_samples_values = [100, 500, 1000,5000]

results = []

for mass_parameter, etta_parameter, number_samples in tqdm(itertools.product(mass_parameter_values, etta_parameter_values, number_samples_values),
                                total=len(mass_parameter_values)*len(etta_parameter_values)*len(number_samples_values)):
    f_err = forbenius_error(nodes_number, mass_parameter, etta_parameter, number_samples)
    results.append({
        "Mass parameter": mass_parameter,
        "Etta parameter": etta_parameter,
        "Number of samples": number_samples,
        "Frobenius_error": f_err
    })

error_df = pd.DataFrame(results)

min(error_df['Frobenius_error'])
#pd.DataFrame.to_csv(error_df,path_or_buf = "C:/Users/Maria/OneDrive - University College London/thesis/code/error_df.csv")


#Heatmap between parameter and error 

for n_samples in number_samples_values:
    plot1 = error_df[error_df["Number of samples"] == n_samples].pivot(index="Mass parameter", columns="Etta parameter", values="Frobenius_error")
    plt.figure(figsize=(6, 4))
    sea.heatmap(plot1, annot=True, cmap="viridis")
    plt.title(f"Error Heatmap (Samples = {n_samples})")
    plt.xlabel("η")
    plt.ylabel("μ")
    plt.tight_layout()
    plt.show()

#3D image between the parameters and the error 

plot2_3D = plt.figure(figsize=(10, 6))
ax = plot2_3D.add_subplot(111, projection='3d')
subset = error_df[error_df["Number of samples"] == 500]
X, Y = np.meshgrid(sorted(subset["Mass parameter"].unique()), sorted(subset["Etta parameter"].unique()))
Z = subset.pivot(index="Etta parameter", columns="Mass parameter", values="Frobenius_error").values
ax.plot_surface(X, Y, Z, cmap=cm.viridis)
ax.set_xlabel('μ (mu)')
ax.set_ylabel('η (eta)')
ax.set_zlabel('Frobenius Error')
ax.set_title("3D Surface Plot of Error (Samples = 500)")
plt.tight_layout()
plt.show()




























#Try to estimate the estimate of Sigma -1 


#We will first estimate the laplacian through the fourier tranform and then the sigma -1
#the approach to estimate L is based on the fourier analytic properties (characteristic function) of the Gaussian 
# distribution - 


#simulate the gaussian free field of the graph a lot of times on the same graph 
# you have all independent GFF - data you will haev access too as a human being - signa hat -1 from those - in this case you also have access to the sigma the true one - are they close enought - how good is the recovering precedure - what is the probability that i am making the mistake 
#check the concentrtion bounds - the mistakes the author are making - the bounds should tell me how big the n should be so you are quarantee that your error is whithin your tolerance 
# when you are a oracle yuo can know the number of n, 















