import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from utils import *

def compute_alpha_admm(X, Y, X_selected, selected_pts_agents, nu, sigma2, W, K, Z, Lambda, beta, node_idx):
    """ Calcul de alpha pour ADMM. """
    
    num_agents = W.shape[0]
    m = len(X_selected)

    # Matrices noyaux
    K_mm = compute_kernel_matrix(X_selected, X_selected)
    K_im = compute_kernel_matrix(selected_pts_agents, X_selected)

    # Construction de A et b
    A = (sigma2 / 5) * K_mm + (nu / 5) * np.eye(m) + K_im.T @ K_im
    b = K_im.T @ Y[selected_pts_agents]

    # Ajustement en fonction des voisins
    for neighbor_idx in range(num_agents):
        if W[node_idx, neighbor_idx] != 0:
            A += beta * np.eye(m)
            b += beta * Z[node_idx, neighbor_idx, :] - Lambda[node_idx, neighbor_idx, :]

    # Résolution pour alpha
    return np.linalg.solve(A, b)

def ADMM(X, Y, X_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta):
    """ Algorithm ADMM """
    
    m = len(X_selected)  # Nombre de points sélectionnés
    Z = np.zeros((a, a, m))
    Lambda = np.zeros((a, a, m))
    alpha = np.zeros((a, m))
    optimality_gaps = []
    alpha_list = []
    alpha_mean_list = []

    for epoch in range(n_epochs):
    
        # Mise à jour de alpha pour chaque agent
        for i in range(a):
            alpha[i, :] = compute_alpha_admm(
                X, Y, X_selected, selected_pts_agents[i], nu, sigma2, W, K, Z, Lambda, beta, i
            )
        
        alpha_list.append(alpha.copy())  # Sauvegarde des valeurs d'alpha
        alpha_mean_list.append(np.mean(alpha, axis=0))  # Moyenne des alpha

        gap = 0
        for i in range(a):
            for j in range(i + 1, a):  # Éviter les mises à jour en double
                if W[i, j] != 0:  
                    # Mise à jour de Z
                    Z[i, j, :] = (alpha[i, :] + alpha[j, :]) / 2
                    Z[j, i, :] = Z[i, j, :]  # Symétrie

                    # Mise à jour de Lambda
                    Lambda[i, j, :] += beta * (alpha[i, :] - Z[i, j, :])
                    Lambda[j, i, :] = -Lambda[i, j, :]  # Symétrie

                    # Calcul de l'optimality gap
                    gap += np.linalg.norm(alpha[i, :] - Z[i, j, :])
        
        optimality_gaps.append(gap)
    
    return alpha, alpha_list, alpha_mean_list, optimality_gaps

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 1
    n_epochs = 10000

    # Génération des données
    x_n = x[:n] 
    y_n = y[:n]

    sel = np.arange(n)
    ind = np.random.choice(sel, m, replace=False)
    x_selected = x[ind]
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)

    # Générer une matrice de poids correcte
    W = W(a)  # Remplace W(a) par une fonction correcte

    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)

    # Calcul de alpha optimal
    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    print(f'Time to compute alpha optimal : {time.time() - start}\n')

    # Exécution d'ADMM
    start = time.time()
    alpha_optim, alpha_list, alpha_mean_list, opt_gaps = ADMM(x_n, y_n, x_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta)
    admm_time = time.time() - start

    # Vérification des dimensions
    print(f"alpha_optim shape: {alpha_optim.shape}")
    print(f"alpha_list shape: {np.array(alpha_list).shape}")

    # Calcul des écarts de norme pour chaque agent
    agent_1 = np.linalg.norm(np.array([alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim[0], axis=1)
    agent_2 = np.linalg.norm(np.array([alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim[1], axis=1)
    agent_3 = np.linalg.norm(np.array([alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim[2], axis=1)
    agent_4 = np.linalg.norm(np.array([alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim[3], axis=1)
    agent_5 = np.linalg.norm(np.array([alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim[4], axis=1)

    # Tracé des résultats
    plt.plot(agent_1, label='Agent 1', color='blue')
    plt.plot(agent_2, label='Agent 2', color='red')
    plt.plot(agent_3, label='Agent 3', color='green')
    plt.plot(agent_4, label='Agent 4', color='orange')
    plt.plot(agent_5, label='Agent 5', color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()