import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from dgd import DGD
from utils import *

def network_newton(X, Y, X_selected, A, nu, sigma2, alpha_star, W, step_size, K, n_epochs=500):
    """
    Network Newton-K Method (NN-K) following the exact algorithm description.
    """
    m = X_selected.shape[0]
    a = len(A)
    # random alpha of shape (a*m, 1)
    alpha = np.zeros((a * m, 1))
    Z = np.kron(W / 3, np.eye(m)) #W_bar
    Z_d = np.diag([Z[i, i] for i in range(a * m)])
    
        # Compute B matrix (weights)
    B = np.eye(a * m) - 2*Z_d + Z
    penality = 0.1
    D = step_size * compute_hessian_alpha(sigma2, nu, Y, X, X_selected, alpha, A, m) + 2 * (np.eye(a * m) - Z_d) 
    
    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []

    for i in range(n_epochs):
        
        g_t = (np.eye(a * m) - Z) @ alpha + step_size * grad_alpha_(sigma2, nu, Y, X, X_selected, alpha, A, m).reshape(a * m, 1)
        
        d = -np.linalg.inv(D) @ g_t
        
        for _ in range(K):
            # exchage d between agents
            d = Z @ d
            d = np.linalg.inv(D) @ (B @ d - g_t)

        # Update alpha
        alpha = alpha + step_size * d
        
        alpha_list_agent.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha.reshape(a, m).mean(axis=0))
        for agent_idx in range(a):
            optimal_gaps[agent_idx].append(np.linalg.norm(alpha.reshape(a, m)[agent_idx] - alpha_star))
    
    alpha_optim = alpha.reshape(a, m).mean(axis=0)
    
    return optimal_gaps, alpha_optim, alpha_list_agent, alpha_mean_list


# def network_newton()


if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5 # Number of data points, number of selected points, number of agents
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 10000
    sigma = 0.5
    K = 10# Order of approximation in NN-K
    step_size = 0.05

    x_n = x[:n]
    y_n = y[:n]
    ind = np.random.choice(list(range(n)), m, replace=False)
    x_selected = np.array([x[i] for i in ind])

    # ind = np.random.choice(list(range(n)), m*a, replace=False)
    N = np.arange(n)
    np.random.shuffle(N)
    A = np.array_split(N, a) 
    
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    W = W_base_bis(a)
    
    print("Running DGD...")
    start = time.time()
    opt_gaps_dgd, alpha_optim_dgd, alpha_list_dgd, alpha_mean_list_dgd = DGD(x_n, y_n, x_selected, a, nu, sigma2, alpha_star, W, step_size, n_epochs)
    end = time.time()
    print(f'DGD time: {end - start}')
    
    print("Running Network Newton...")
    start = time.time()
    opt_gaps_nn, alpha_nn, alpha_list_nn, alpha_mean_list_nn = network_newton(x_n, y_n, x_selected, A, nu, sigma2, alpha_star, W, step_size, K, n_epochs)
    end = time.time()
    print(f'Network Newton time: {end - start}')
    
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
    plt.plot(agent_5, label='DGD agents', color='blue')
    plt.plot(agent_1_nn,  label=f'NN-{K} agent', color='red')
    plt.plot(agent_2_nn,  color='red')
    plt.plot(agent_3_nn,  color='red')
    plt.plot(agent_4_nn,  color='red')
    plt.plot(agent_5_nn,  color='red')
    
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f'Optimality gap comparison between DGD and NN-K with K = {K}\nstep_size = {step_size}')
    plt.legend(loc='lower left', ncols=1, fontsize=8)
    plt.tight_layout()
    #plt.savefig('opt_gaps_DGD_vs_NN.png', bbox_inches='tight')
    plt.grid()
    plt.show()
