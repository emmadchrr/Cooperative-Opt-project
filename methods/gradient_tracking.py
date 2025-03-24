import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
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

    W_bar = np.kron(W / a, np.eye(m))   # Matrice de consensus

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