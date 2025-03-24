import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from methods.dgd import DGD
from utils import *

def network_newton(X, Y, X_selected, a, nu, sigma2, alpha_star, W, step_size, K, n_epochs=500):
    """
    Network Newton-K Method (NN-K) following the exact algorithm description.
    """
    m = X_selected.shape[0]
    alpha = np.zeros((a * m, 1))
    W_bar = np.kron(W / 3, np.eye(m))
    
    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    Kim = compute_kernel_matrix(X, X_selected)

    for _ in range(n_epochs):
        # Compute gradient
        grad = grad_alpha(sigma2, nu, Y, X, X_selected, alpha, a, m)
        grad = grad.reshape(a * m, 1)
        # Compute B matrix (weights)
        B = W_bar
        
        D = np.kron(np.eye(a), compute_local_Hessian(sigma2, Kmm, Kim, nu, a))
        d = -np.linalg.inv(D) @ grad
        
        # NN-K iterations
        for k in range(K):
            d_new = np.zeros_like(d)
            for i in range(a * m):
                neighbors = [j for j in range(a * m) if B[i, j] != 0]
                if D[i, i] != 0:
                    d_new[i] = (1 / D[i, i]) * (sum(B[i, j] * d[j] for j in neighbors) - grad[i])
                else:
                    d_new[i] = 0  # ou une autre valeur appropriée   
            d = d_new
        
        # Update alpha
        alpha = alpha + step_size * d
        
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
    n_epochs = 10000
    sigma = 0.5
    K = 5  # Order of approximation in NN-K
    step_size = 0.002

    x_n = x[:n]
    y_n = y[:n]
    ind = np.random.choice(range(n), m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    W = W(a)
    
    print("Running DGD...")
    start = time.time()
    opt_gaps_dgd, alpha_optim_dgd, alpha_list_dgd, alpha_mean_list_dgd = DGD(x_n, y_n, x_selected, a, nu, sigma2, alpha_star, W, step_size, n_epochs)
    end = time.time()
    print(f'DGD time: {end - start}')
    
    print("Running Network Newton...")
    start = time.time()
    opt_gaps_nn, alpha_nn, alpha_list_nn, alpha_mean_list_nn = network_newton(x_n, y_n, x_selected, a, nu, sigma2, alpha_star, W, step_size, K, n_epochs)
    end = time.time()
    print(f'Network Newton time: {end - start}')
    
    # Data visualization
    Y_dgd = np.linalg.norm(alpha_list_dgd - alpha_star, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
    agent_1 = np.linalg.norm(np.array(
        [alpha_list_dgd[i][0] for i in range(len(alpha_list_dgd))]) - alpha_star, axis=1)
    agent_2 = np.linalg.norm(np.array(
        [alpha_list_dgd[i][1] for i in range(len(alpha_list_dgd))]) - alpha_star, axis=1)
    agent_3 = np.linalg.norm(np.array(
        [alpha_list_dgd[i][2] for i in range(len(alpha_list_dgd))]) - alpha_star, axis=1)
    agent_4 = np.linalg.norm(np.array(
        [alpha_list_dgd[i][3] for i in range(len(alpha_list_dgd))]) - alpha_star, axis=1)
    agent_5 = np.linalg.norm(np.array(
        [alpha_list_dgd[i][4] for i in range(len(alpha_list_dgd))]) - alpha_star, axis=1)

    Y_nn = np.linalg.norm(alpha_list_nn - alpha_star, axis=1)
    # unpack the list of alpha to get for each agent the evolution of alpha
    agent_1_nn = np.linalg.norm(np.array(
        [alpha_list_nn[i][0] for i in range(len(alpha_list_nn))]) - alpha_star, axis=1)
    agent_2_nn = np.linalg.norm(np.array(
        [alpha_list_nn[i][1] for i in range(len(alpha_list_nn))]) - alpha_star, axis=1)
    agent_3_nn = np.linalg.norm(np.array(
        [alpha_list_nn[i][2] for i in range(len(alpha_list_nn))]) - alpha_star, axis=1)
    agent_4_nn = np.linalg.norm(np.array(
        [alpha_list_nn[i][3] for i in range(len(alpha_list_nn))]) - alpha_star, axis=1)
    agent_5_nn = np.linalg.norm(np.array(
        [alpha_list_nn[i][4] for i in range(len(alpha_list_nn))]) - alpha_star, axis=1)

    plt.plot(agent_1, color='blue')
    plt.plot(agent_2, color='blue')
    plt.plot(agent_3, color='blue')
    plt.plot(agent_4, color='blue')
    plt.plot(agent_5, label='DGD', color='blue')
    plt.plot(agent_1_nn, label=f'NN - {K}\n', color='red')
    plt.plot(agent_2_nn, color='red')
    plt.plot(agent_3_nn, color='red')
    plt.plot(agent_4_nn, color='red')
    plt.plot(agent_5_nn, color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f'Optimality gap comparison between DGD and NN-K with K = {K}\nstep_size = {step_size}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.savefig('opt_gaps_DGD_vs_NN.png', bbox_inches='tight')
    plt.grid()
    plt.show()
