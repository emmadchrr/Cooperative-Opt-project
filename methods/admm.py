import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from utils import *

def ADMM(X, Y, X_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta):
    """
    ADMM Implementation
    """

    num_selected = len(X_selected)  # Number of selected points

    # Define neighbors for each agent based on the adjacency matrix
    neighbors = [[j for j in range(a) if j != i and W[i, j] > 0.0001] for i in range(a)]

    # Compute kernel matrices
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    KimKim = [compute_kernel_matrix(X[selected_pts_agents[i]], X_selected).T @ 
              compute_kernel_matrix(X[selected_pts_agents[i]], X_selected) for i in range(a)]
    yKnm = [compute_kernel_matrix(X[selected_pts_agents[i]], X_selected).T @ Y[selected_pts_agents[i]] for i in range(a)]

    # Initialize dual variables and auxiliary variables
    lambda_k = {}  # Lagrange multipliers
    y_k = {}  # Auxiliary variables for consensus

    for i in range(a):
        for j in neighbors[i]:
            lambda_k[i, j] = np.zeros(num_selected)  # Initialize Lagrange multipliers
            if i < j:
                s = np.random.rand(num_selected)  # Random initialization
                y_k[i, j] = s
                y_k[j, i] = s  # Ensure symmetry

    iteration = 0
    deviation_alpha_star = []  # Track deviation from the optimal alpha_star
    alpha_list = []
    alpha_mean_list = []

    # Compute the reference optimal solution alpha_star
    Knm = compute_kernel_matrix(X, X_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, Y, sigma2, nu)

    while iteration <= n_epochs:
        iteration += 1

        # Update alpha_k for each agent
        alpha_k = [
            np.linalg.solve(
                (nu / a) * np.identity(num_selected) +
                (sigma2 / a) * Kmm +
                KimKim[i] +
                sum(beta * np.identity(num_selected) for j in neighbors[i]),  # Adjust for neighbors

                yKnm[i] + sum(beta * y_k[i, j] - lambda_k[i, j] for j in neighbors[i])
            )
            for i in range(a)
        ]
        alpha_k = np.array(alpha_k)  # Convert to NumPy array

        # Store alpha values for tracking
        alpha_list.append(alpha_k.copy())  
        alpha_mean_list.append(np.mean(alpha_k, axis=0))  # Compute mean alpha

        # Compute deviation from alpha_star
        deviation_alpha_star.append([np.linalg.norm(alpha_k[i] - alpha_star) for i in range(a)])

        # Update auxiliary variables y_k and Lagrange multipliers lambda_k
        for i in range(a):
            for j in neighbors[i]:
                y_k[i, j] = 0.5 * (alpha_k[i] + alpha_k[j])  # Average of neighbor alphas
                lambda_k[i, j] += beta * (alpha_k[i] - y_k[i, j])  # Update Lagrange multipliers

    return alpha_k, alpha_list, alpha_mean_list, np.array(deviation_alpha_star)

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 10000

    # Data Generation
    x_n = x[:n] 
    y_n = y[:n]

    sel = np.arange(n)
    ind = np.random.choice(sel, m, replace=False)
    x_selected = x[ind]
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

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    print(f'Time to compute alpha optimal : {time.time() - start}\n')

    start = time.time()
    alpha_optim, alpha_list, alpha_mean_list, opt_gaps = ADMM(x_n, y_n, x_selected, selected_pts_agents, a, nu, sigma2, n_epochs, W, K, beta)
    admm_time = time.time() - start

    print(f"alpha_optim shape: {alpha_optim.shape}")
    print(f"alpha_list shape: {np.array(alpha_list).shape}")

    agent_1 = np.linalg.norm(np.array([alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optim[0], axis=1)
    agent_2 = np.linalg.norm(np.array([alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optim[1], axis=1)
    agent_3 = np.linalg.norm(np.array([alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optim[2], axis=1)
    agent_4 = np.linalg.norm(np.array([alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optim[3], axis=1)
    agent_5 = np.linalg.norm(np.array([alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optim[4], axis=1)

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
