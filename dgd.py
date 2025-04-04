import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from utils import *
import time

def DGD(X, Y, X_selected, A, nu, sigma2, alpha_star, W, step_size, n_epochs=500):
    """
    Decentralized Gradient Descent (DGD) optimisé.
    """
    m = X_selected.shape[0]
    a = len(A)
    
    # Initialization of alpha
    alpha = np.zeros((a * m, 1))
    
    #Normalized Weight Matrix
    W_bar = np.kron(W / 3, np.eye(m))
        
    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []
    
    for _ in range(n_epochs):
        grad = grad_alpha(sigma2, nu, Y, X, X_selected, alpha, A, m)
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
    step_size = 0.0009

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
    W = W_base_bis(a)
    # W = fully_connected_graph(a)
    #W = linear_graph(a)
    #W = small_world_graph(a)
    K = compute_kernel_matrix(x_n, x_n)
    N = np.arange(n)
    np.random.shuffle(N)
    A = np.array_split(N, a)


    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')
    start = time.time()

    opt_gaps, alpha_optim, alpha_list, alpha_mean_list = DGD(
        x_n, y_n, x_selected, A, nu, sigma2, alpha_optimal, W, step_size, n_epochs=n_epochs)
    end = time.time()
    print(f'alpha optimal with DGD : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with DGD : {end - start}')

    plot_optimality_gaps(
        alpha_list, 
        alpha_optimal, 
        log_scale=True, 
    )
