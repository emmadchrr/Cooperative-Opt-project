import sys
import os
from utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

    
def solve_alpha(x, y, x_selected, A, sigma, nu, adj_matrix, lamb):
    """
    Computes the alpha values for communication step in dual decomposition."
    """

    a = len(A)
    num_selected = len(x_selected)
    
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    # Initialize list to store alpha values for each agent
    alpha_values = []

    for agent in range(a):
        Kim = compute_kernel_matrix(x[A[agent]], x_selected)
        A_matrix = sigma**2 * Kmm + np.eye(num_selected) * nu + Kim.T @ Kim
        b_vector = Kim.T @ y[A[agent]] # Compute the right-hand side vector b

        # Adjust b_vector based on Lagrange multipliers (lambda) from neighboring agents
        for neighbor_idx in range(a):
            if adj_matrix[agent, neighbor_idx] != 0:
                if agent > neighbor_idx:
                    b_vector -= lamb[agent, neighbor_idx, :]
                else:
                    b_vector += lamb[neighbor_idx, agent, :]

        alpha_values.append(np.linalg.solve(A_matrix, b_vector))

    return np.array(alpha_values)

def dual_decomposition(x, y, X_selected, A, nu, sigma2, W, step_size, nb_epochs=1000, lamb0=0.0):
    """
    Performs dual decomposition for distributed kernel-based learning.
    """
    # Construct the communication graph (binary adjacency matrix)
    communication_graph = (W > 0).astype(int)
    
    num_selected = len(X_selected)
    a = len(A)
    
    # Ensure diagonal of graph is zero (no self-connections)
    for agent in range(a):
        communication_graph[agent, agent] = 0

    # Initialize Lagrange multipliers
    lambda_ij = lamb0 * np.ones((a, a, num_selected))

    # Lists to track alpha values across iterations
    alpha_mean_list = []
    alpha_list_agent = []

    for _ in tqdm(range(nb_epochs)):
        # Compute optimal alpha values for each agent
        alpha_optim = solve_alpha(
            x, y, X_selected, A, sigma2, nu, 
            communication_graph, lambda_ij
        )

        # Update Lagrange multipliers for connected agents
        for agent_i in range(a):
            for agent_j in range(agent_i):  # Only update for i > j
                lambda_ij[agent_i, agent_j, :] += step_size * (alpha_optim[agent_i, :] - alpha_optim[agent_j, :])

        # Track mean alpha values and per-agent alphas over iterations
        alpha_mean_list.append(alpha_optim.mean(axis=0))
        alpha_list_agent.append(alpha_optim)

    # Compute the final optimized alpha as the mean across agents
    alpha_optim = np.mean(alpha_optim.reshape(a, num_selected), axis=0)

    return alpha_optim, alpha_list_agent, alpha_mean_list

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 1
    n_epochs = 100
    step_size = 0.01

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)


    W = fully_connected_graph(a)

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
    alpha_optim, alpha_list, alpha_mean_list = dual_decomposition(x_n, y_n, x_selected, A, nu, sigma2, W, step_size=step_size, nb_epochs=n_epochs, lamb0=0.0)
    end = time.time()
    print(f'alpha optimal with dual decomposition : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with dual decomposition : {end - start}')

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
    #plt.savefig('opt_gaps_dual_dec_with_agents_scalelog.png', bbox_inches='tight')
    plt.grid()
    plt.show()
    