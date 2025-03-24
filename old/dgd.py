import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'methods.utils')))
from utils import *


def DGD(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size):
    """
    Decentralized gradient descent algorithm with multiple agents.
    """
    m = X_selected.shape[0]
    n = X.shape[0]

    a_data_idx = np.array_split(np.random.permutation(n), a)
    W = np.kron(W, np.ones((m, m)))
    alpha = np.ones((m*a, 1))
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    opt_gaps = [[] for _ in range(a)]
    alphas = []

    for _ in range(n_epochs):
        alphas.append(alpha)
        gradients = np.zeros((m * a, 1))

        for agent_idx, data_idx in enumerate(a_data_idx):
            X_local = X[data_idx]
            y_local = Y[data_idx].reshape(-1, 1)
            K_im = compute_kernel_matrix(X_local, X_selected)
            local_gradient = compute_local_gradient(alpha[m * agent_idx: m * (agent_idx + 1)], sigma2, Kmm, y_local, K_im, nu, a)
            gradients[m * agent_idx: m * (agent_idx + 1)] = local_gradient
        
        alpha = W @ alpha - step_size * gradients
        
        for agent_idx in range(a):
            alpha_agent = alpha[m * agent_idx: m * (agent_idx + 1)].reshape(-1, 1)
            optimality_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            opt_gaps[agent_idx].append(optimality_gap)
        
    return opt_gaps, alphas


if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 100
    sigma = 0.5
    step_size = 0.002

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    W = np.zeros((a,a))
    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)
    lr = 0.01

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')

    start = time.time()
    opt_gaps, alphas = DGD(
        x_n, y_n, x_selected, a, nu, sigma2, n_epochs, alpha_optimal, W, 0.01)
    end = time.time()
    dgd_time = end - start
    
    for agent_idx, optimality_gaps in enumerate(opt_gaps):
        plt.plot(range(n_epochs), optimality_gaps, label=f"Agent {agent_idx + 1}")

    plt.xlabel('Number of iterations')
    plt.ylabel('Optimality Gap')
    plt.title(f'DGD Convergence, Step size : {step_size}')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show()