#ADMM 

import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import *

def compute_alpha_admm(X, Y, X_selected, selected_pts_agents, nu, sigma2, W, K, Z, Lambda, beta, node_idx):
    
    num_agents = W.shape[0]
    m = len(X_selected)

    # Compute kernel matrices
    K_mm = compute_kernel_matrix(X_selected, X_selected)
    K_im = compute_kernel_matrix(selected_pts_agents, X_selected)

    # Construct matrix A and vector b
    A = (sigma2 / 5) * K_mm + (nu / 5) * np.eye(m) + K_im.T @ K_im
    b = K_im.T @ Y[selected_pts_agents]

    # Adjust A and b based on adjacency relationships
    for neighbor_idx in range(num_agents):
        if W[node_idx, neighbor_idx] != 0:
            A += beta * np.eye(m)
            b += beta * Z[node_idx, neighbor_idx, :] - Lambda[node_idx, neighbor_idx, :]

    # Solve for alpha
    return np.linalg.solve(A, b)

def ADMM(X, Y, X_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta):
    """
    Alternating Direction Method of Multipliers (ADMM) algorithm with multiple agents.
    """
    m = len(X_selected)  # Nombre de points sélectionnés
    
    # Initialisation des variables duales et primales
    Z = np.zeros((a, a, m))
    Lambda = np.zeros((a, a, m))
    alpha = np.zeros((a, m))
    optimality_gaps = []

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        for i in range(a):
            # Mise à jour de alpha pour chaque agent
            alpha[i, :] = compute_alpha_admm(
                X, Y, X_selected, selected_pts_agents[i], nu, sigma2, W, K, Z, Lambda, beta, i
            )
        
        gap = 0
        for i in range(a):
            for j in range(a):
                if W[i, j] != 0:  # Vérifier la connectivité entre les agents
                    # Mise à jour de Z
                    Z[i, j, :] = (alpha[i, :] + alpha[j, :]) / 2
                    # Mise à jour de Lambda
                    Lambda[i, j, :] +=  beta * (alpha[i, :] - Z[i, j, :])
                    
                    # Calcul de l'optimality gap
                    gap += np.linalg.norm(alpha[i, :] - Z[i, j, :])
        optimality_gaps.append(gap)
    
    return alpha, optimality_gaps

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 1
    n_epochs = 100

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    W = np.ones((a,a))
    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')

    # Run ADMM
    start = time.time()
    alpha, opt_gaps = ADMM(x_n, y_n, x_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta)
    admm_time = time.time() - start

    plt.figure(figsize=(10, 6))
    plt.plot(opt_gaps)
    plt.xlabel('Iterations')
    plt.ylabel('Optimality Gap')
    plt.title('Optimality Gap of ADMM')
    plt.grid(True)
    plt.savefig('optimality_gaps_admm.png')
    plt.show()
    print(f'Time to run ADMM : {admm_time}\n')
    print(f'Optimality gap : {opt_gaps[-1]}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')
