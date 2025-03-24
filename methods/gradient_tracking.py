import sys
import os
from utils import *
import numpy as np

def gradient_tracking(x, y, x_selected, a, nu, sigma2, alpha_star, W, lr, n_epochs = 500):
    """
    Implémente l'algorithme de Gradient Tracking (GT).
    
    Paramètres :
    - x, y : Données globales
    - x_selected : Points sélectionnés pour approximation de Nyström
    - a : Nombre d'agents
    - nu : Paramètre de régularisation
    - sigma2 : Hyperparamètre du bruit
    - alpha_star : Solution optimale pour comparaison
    - W : Matrice de consensus
    - lr : Learning rate
    - n_epochs : Nombre d'itérations

    Retourne :
    - optimal_gaps : Liste contenant l'évolution de ||alpha^i - alpha_star|| pour chaque agent.
    - alpha_optim : Dernière valeur de alpha moyenne sur les agents.
    - alpha_list_agent : Liste des valeurs de alphas à chaque itération pour chaque agent.
    - alpha_mean_list : Liste des moyennes de alphas sur les agents à chaque itération.
    """

    m = x_selected.shape[0]

    alpha = np.zeros((a * m, 1))        # Paramètres locaux des agents (a*m, 1)
    g = np.zeros((a * m, 1))            # Terme de suivi du gradient (a*m, 1)

    W_bar = np.kron(W, np.eye(m))   # Matrice de consensus

    Kmm = compute_kernel_matrix(x_selected, x_selected)

    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []

    # initialisation du gradient
    grad_old = grad_alpha(sigma2, nu, y, x, x_selected, alpha, a, m).reshape((a * m, 1))  # (a, m)
    g = grad_old.copy()

    for epoch in range(n_epochs):
        # Mise à jour de alpha_i en utilisant g
        alpha = W_bar @ alpha - lr * g

        # Calcul du nouveau gradient
        grad_new = grad_alpha(sigma2, nu, y, x, x_selected, alpha, a, m).reshape((a * m, 1))  # (a, m)

        # Mise à jour de g^i (suivi du gradient)
        g = (W_bar @ g + (grad_new - grad_old))

        grad_old = grad_new
    
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
    W = fully_connected_graph(a)
    #W = linear_graph(a)
    #W = small_world_graph(a)

    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)
    step_size = 0.002

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')
    start = time.time()

    opt_gaps, alpha_optim, alpha_list, alpha_mean_list = gradient_tracking(
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
    
 