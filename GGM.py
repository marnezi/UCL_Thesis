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

# Step 1: Create a Erdos Reyni graph to test 
nodes_number = 10 # Laplacian_matrix.shape[1] # number of nodes 
transition_prob = 0.6 # probability of # of edges per node (we keep it high so we have less disconnected samples)

np.log(10)/10

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

number_samples = 70000
mass_parameter = 10# we keep this fixed if it is too smaller than one the eigenvalues of sigma will be close to zero and make it ill-conditioned 
etta_parameter = 1 # smoothing parameter 

fixed_node = 0 # at least one node equal to zero Dirichlet boundary condition 

#Check 
((nodes_number**4)*np.log(nodes_number))/(transition_prob**2)
number_samples
number_samples> (nodes_number**4)*np.log(nodes_number)*(transition_prob**(-2))


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
    print("⚠️ Warning: eta I - L_hat is nearly singular!")

    
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
        # estimating the empirical characteristic function phi 
        phi = np.mean(np.exp(1j * np.sum(y_samples * (x_samples + t), axis=1)))
        return phi
    
    #Create the etta Laplacian matrix from the fourier analysis (smoothed laplacian)

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
                # bias is added to the final estimator (because of the nonlinear tranformation/log) - jensens inequality 
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















