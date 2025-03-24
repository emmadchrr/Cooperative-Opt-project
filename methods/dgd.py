import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys
import os
from utils import *
import time

def DGD(X, Y, X_selected, a, nu, sigma2, alpha_star, W, step_size, n_epochs=500):
    """
    Decentralized Gradient Descent (DGD) optimisé.
    """
    m = X_selected.shape[0]
    
    # Initialisation de alpha
    alpha = np.zeros((a * m, 1))
    
    # Matrice de poids normalisée
    W_bar = np.kron(W, np.eye(m))
    
    # Calcul du noyau entre les points sélectionnés
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    
    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []
    
    for _ in range(n_epochs):
        grad = grad_alpha(sigma2, nu, Y, X, X_selected, alpha, a, m)
        alpha = W_bar @ alpha - step_size * grad.reshape(a * m, 1)
        alpha_list_agent.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha.reshape(a, m).mean(axis=0))
        for agent_idx in range(a):
            optimal_gaps[agent_idx].append(np.linalg.norm(alpha.reshape(a, m)[agent_idx] - alpha_star))
    
    alpha_optim = alpha.reshape(a, m).mean(axis=0)
    
    return optimal_gaps, alpha_optim, alpha_list_agent, alpha_mean_list


if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 100
    sigma = 0.5

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    #W = np.ones((a, a))
    #W = W_base(a)
    #W = fully_connected_graph(a)
    #W = linear_graph(a)
    W = small_world_graph(a)
    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)
    step_size = 0.002

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')
    start = time.time()

    opt_gaps, alpha_optim, alpha_list, alpha_mean_list = DGD(
        x_n, y_n, x_selected, a, nu, sigma2, alpha_optimal, W, step_size, n_epochs=10000)
    end = time.time()
    print(f'alpha optimal with DGD : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with DGD : {end - start}')
    # print(f'Total iterations : {tot_ite}\n')

    # Data visualization
    Y = np.linalg.norm(alpha_list - alpha_optim, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
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

    plt.plot(agent_1, label='Agent 1', color='blue')
    plt.plot(agent_2, label='Agent 2', color='red')
    plt.plot(agent_3, label='Agent 3', color='green')
    plt.plot(agent_4, label='Agent 4', color='orange')
    plt.plot(agent_5, label='Agent 5', color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    #plt.savefig('opt_gaps_DGD_with_agents_scalelog.png', bbox_inches='tight')
    plt.grid()
    plt.show()
    
 