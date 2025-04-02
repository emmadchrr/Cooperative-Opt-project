import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import pickle
from utils import compute_kernel_matrix, compute_alpha_star
import pdb

def directed_graph(a, density=0.6):
    """
    Génère un graphe dirigé connecté avec `a` nœuds.
    """
    while True:
        G = nx.erdos_renyi_graph(a, density, directed=True)
        if nx.is_strongly_connected(G):  # Vérifie la connectivité
            break
    
    return nx.to_numpy_array(G, nodelist=range(a))

def directed_push_sum(n, adjacency_matrix, initial_values, max_iter=1000):
    """
    Implémentation du Push-Sum.
    """
    values = initial_values.copy()  # alpha_k^j
    weights = np.ones(n)  # φ_k^j initialisé à 1
    history = []
    epsilon = 1e-10

    out_degrees = adjacency_matrix.sum(axis=1)  #
    out_degrees[out_degrees == 0] = 1  

    for k in range(max_iter):
        print(f"Itération {k}/{max_iter}")
        if k % 100 == 0:  # Pour ne pas imprimer trop souvent
            print("Valeurs actuelles:", values)
        
        normalized_values = values / (1 + out_degrees)[:, None]
        normalized_weights = weights / (1 + out_degrees)

        # Envoi des valeurs aux voisins
        new_values = normalized_values.T @ adjacency_matrix 
        new_weights = normalized_weights @ adjacency_matrix  

        new_values += normalized_values[:, 0]  
        new_weights += normalized_weights  
        # Calcul du ratio z_k+1^i = alpha_k+1^i / φ_k+1^i
        new_values = new_values / (new_weights + epsilon) 
        
        values, weights = new_values, new_weights
        history.append(values)

    return values, history

def run_experiment_with_push_sum(x, y, X_selected, selected_pts_agents, W, K, sigma, nu, beta, n_epochs, max_iter=1000):
    """
    Exécute le protocole Push-Sum et calcule l'écart à l'optimalité.
    """
    n = len(x)
    a = len(W)
    
    # Chaque agent commence avec la moyenne de ses propres données
    initial_values = np.zeros(a)
    for i in range(a):
        agent_indices = selected_pts_agents[i]
        initial_values[i] = np.mean(y_n[agent_indices])

    adjacency_matrix = np.array(W)

    # Exécution du protocole Push-Sum
    print('Execution de Push-Sum...')
    start = time.time()
    final_values, history = directed_push_sum(a, adjacency_matrix, initial_values, max_iter=max_iter)
    end = time.time()
    print(f'Temps d’exécution du Push-Sum: {end - start} secondes')

    # Calcul de la valeur de consensus
    consensus_value = np.mean(final_values)
    optimality_gaps = [np.abs(final_values - consensus_value).mean()]

    # Tracé de la convergence
    history = np.array(history)
    convergence = np.abs(history - consensus_value).mean(axis=1)

    plt.plot(convergence)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Gap with consensus')
    plt.title('Convergence of Push-Sum')
    plt.grid(True)
    plt.show()

    return final_values, optimality_gaps, history

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 1000
    sigma = 0.5

    # Sélection des points et calcul des matrices noyau
    sel = np.arange(n)
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    x_n = x[:n]
    y_n = y[:n]
    
    print('Calcul de alpha_star...')
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    print("Dimensions de Kmm:", Kmm.shape)
    print("Dimensions de Knm:", Knm.shape)

    # Génération d'un graphe dirigé et connecté
    start = time.time()
    print('Génération du graphe dirigé...')
    W = directed_graph(a)
    end = time.time()
    print("Matrice d'adjacence W:\n", W)
    print(f'Temps de génération du graphe: {end - start} secondes')

    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)
    print(f'Exécution de Push-Sum avec {a} agents...')
    # Exécution de Push-Sum
    final_values, optimality_gaps, history = run_experiment_with_push_sum(
        x_n, y_n, x_selected, selected_pts_agents, W, K, sigma, nu, beta, n_epochs
    )

    print(f'Écart final à l\'optimalité: {optimality_gaps[0]}')



