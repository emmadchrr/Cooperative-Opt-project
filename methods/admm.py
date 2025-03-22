import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from utils import *

def ADMM(X, Y, X_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta):
    """
    Implémentation de l'ADMM alignée sur la première version fournie.
    
    Arguments :
    - X, Y : Données d'entrée et sorties
    - X_selected : Points sélectionnés pour l'apprentissage
    - selected_pts_agents : Liste des indices de points pour chaque agent
    - a : Nombre d'agents
    - nu, sigma2 : Paramètres du noyau
    - n_epochs : Nombre d'itérations
    - W : Matrice de poids du graphe
    - K : Matrice de noyau complète
    - beta : Paramètre de régularisation

    Retourne :
    - alpha_k : Solution finale
    - alpha_list : Historique des alpha_k
    - alpha_mean_list : Moyenne des alpha_k à chaque itération
    - ecart_alpha_star : Liste des écarts à alpha_star
    """

    m = len(X_selected)  # Nombre de points sélectionnés

    # Définition des voisins pour chaque agent
    voisins = [[j for j in range(a) if j != i and W[i, j] > 0.0001] for i in range(a)]

    # Calcul des noyaux
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    KimKim = [compute_kernel_matrix(X[selected_pts_agents[i]], X_selected).T @ compute_kernel_matrix(X[selected_pts_agents[i]], X_selected) for i in range(a)]
    yKnm = [compute_kernel_matrix(X[selected_pts_agents[i]], X_selected).T @ Y[selected_pts_agents[i]] for i in range(a)]

    # Initialisation des variables duales et auxiliaires
    lambda_k = dict()
    y_k = dict()
    
    for i in range(a):
        for j in voisins[i]:
            lambda_k[i, j] = np.zeros(m)  # Initialisation des multiplicateurs de Lagrange
            if i < j:
                s = np.random.rand(m)  # Initialisation aléatoire
                y_k[i, j] = s
                y_k[j, i] = s  # Assurer la symétrie

    nb_iter = 0
    ecart_alpha_star = []  # Liste des écarts à alpha_star
    alpha_list = []
    alpha_mean_list = []

    # Calcul de alpha_star (référence optimale)
    Knm = compute_kernel_matrix(X, X_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, Y, sigma2, nu)

    while nb_iter <= n_epochs:
        nb_iter += 1

        # Mise à jour des alpha_k pour chaque agent
        alpha_k = [
            np.linalg.solve(
                (nu / a) * np.identity(m)
                + (sigma2 / a) * Kmm
                + KimKim[i]
                + sum(beta * np.identity(m) for j in voisins[i]),  # Ajustement pour les voisins
                
                yKnm[i] + sum(beta * y_k[i, j] - lambda_k[i, j] for j in voisins[i])
            )
            for i in range(a)
        ]
        alpha_k = np.array(alpha_k)  # Convertir en un tableau NumPy


        alpha_list.append(alpha_k.copy())  # Sauvegarde des valeurs d'alpha
        alpha_mean_list.append(np.mean(alpha_k, axis=0))  # Moyenne des alpha
        
        # Calcul de l'écart à alpha_star
        ecart_alpha_star.append([np.linalg.norm(alpha_k[i] - alpha_star) for i in range(a)])

        # Mise à jour des variables auxiliaires y_k et lambda_k
        for i in range(a):
            for j in voisins[i]:
                y_k[i, j] = 0.5 * (alpha_k[i] + alpha_k[j])  # Moyenne des alphas des voisins
                lambda_k[i, j] += beta * (alpha_k[i] - y_k[i, j])  # Mise à jour des multiplicateurs de Lagrange

    return alpha_k, alpha_list, alpha_mean_list, np.array(ecart_alpha_star)

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 0.1
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
