import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys
import os
from utils import *
import time

def DGD_DP(X, Y, X_selected, a, nu, sigma2, alpha_star, W, epsilon, n_epochs=500):
    """
    Differentially Private Decentralized Gradient Descent (DP-DGD)
    """
    m = X_selected.shape[0]
    n = X.shape[0]
    
    # Initialize alpha randomly for each agent
    alpha = [np.random.rand(m) for _ in range(a)]
    
    # Create data partition for each agent
    A = np.array_split(np.random.permutation(n), a)
    
    # Kernel computation
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    Knm = compute_kernel_matrix(X, X_selected)
    
    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []
    
    for iter in range(1, n_epochs+1):
        # Compute DP parameters that decay with iteration
        mu_k = 0.002 / (1 + 0.001 * iter)
        gamma_k = 1 / (1 + 0.001 * iter**0.9)
        nu_k = (0.01 / epsilon) * (1 / (1 + 0.001 * iter**0.1))
        
        # Add Laplace noise
        ksi_k = [np.random.laplace(0, np.sqrt(nu_k/2), m) for _ in range(a)]
        
        # DP-DGD update
        new_alpha = []
        for i in range(a):
            # Gradient computation for agent i's data
            grad = (1/a)*sigma2*Kmm.dot(alpha[i]) + nu*(1/a)*alpha[i] + \
                   sum([Knm[j]*(Knm[j].dot(alpha[i])) - Y[j]*Knm[j] for j in A[i]])
            
            # Consensus term with noise
            consensus = sum([W[i,j] * (alpha[j] + ksi_k[j] - alpha[i]) for j in range(a) if j != i])
            
            # Update
            new_alpha_i = alpha[i] + gamma_k * consensus - mu_k * grad
            new_alpha.append(new_alpha_i)
        
        alpha = new_alpha
        alpha_list_agent.append(alpha.copy())
        alpha_mean_list.append(np.mean(alpha, axis=0))
        
        # Track optimality gaps
        for agent_idx in range(a):
            optimal_gaps[agent_idx].append(np.linalg.norm(alpha[agent_idx] - alpha_star))
    
    alpha_optim = np.mean(alpha, axis=0)
    
    return optimal_gaps, alpha_optim, alpha_list_agent, alpha_mean_list


if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 10000
    sigma = 0.5
    noise_scales = [0.1, 1, 10]
    colors = ['blue', 'orange', 'green']

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)

    W = W_base(a)
    
    for noise in noise_scales:
        opt_gaps, alpha_optim, alpha_list, alpha_mean_list = DGD_DP(
            x_n, y_n, x_selected, a, nu, sigma2, alpha_star, W, noise, n_epochs=n_epochs)
    

        agent_1 = np.linalg.norm(np.array(
            [alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
        agent_2 = np.linalg.norm(np.array(
            [alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
        agent_3 = np.linalg.norm(np.array(
            [alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
        agent_4 = np.linalg.norm(np.array(
            [alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim, axis=1)
        agent_5 = np.linalg.norm(np.array(
            [alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim, axis=1)

        plt.plot(agent_1, label=f'$\\epsilon$ = {noise}', color=colors[noise_scales.index(noise)])
        plt.plot(agent_2, color=colors[noise_scales.index(noise)])
        plt.plot(agent_3, color=colors[noise_scales.index(noise)])
        plt.plot(agent_4, color=colors[noise_scales.index(noise)])
        plt.plot(agent_5, color=colors[noise_scales.index(noise)])
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title('Optimality gap for each agent with different noise scales')
    #plt.savefig('opt_gaps_DPDGD.png', bbox_inches='tight')
    plt.grid()
    plt.show()