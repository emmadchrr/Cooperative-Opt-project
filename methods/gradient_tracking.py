import sys
import os
from utils import *
import numpy as np

def gradient_tracking(x, y, X_selected, A, nu, sigma2, alpha_star, W, step_size, n_epochs = 500):
    """
    Implement the GT algorithm
    """
    a = len(A)
    m = X_selected.shape[0]
    alpha = np.zeros((a * m, 1))        # local parameters of the agents
    g = np.zeros((a * m, 1))            # gradient tracking term (a*m, 1)
    W_bar = np.kron(W, np.eye(m))   # Consensus matrix


    optimal_gaps = [[] for _ in range(a)]
    alpha_list_agent = []
    alpha_mean_list = []

    # initialization of the gradient
    grad_old = grad_alpha(sigma2, nu, y, x, X_selected, alpha, A, m).reshape((a * m, 1))  # (a, m)
    g = grad_old.copy()

    for epoch in range(n_epochs):
        # update alpha_i using g
        alpha = W_bar @ alpha - step_size * g

        grad_new = grad_alpha(sigma2, nu, y, x, X_selected, alpha, A, m).reshape((a * m, 1))  # (a, m)

        # update g^i
        g = (W_bar @ g + (grad_new - grad_old))

        grad_old = grad_new
    
        alpha_list_agent.append(alpha.reshape(a, m))
        alpha_mean_list.append(alpha.reshape(a, m).mean(axis=0))

        for agent in range(a):
            optimal_gaps[agent].append(np.linalg.norm(alpha.reshape(a, m)[agent] - alpha_star))

    alpha_optim = alpha.reshape(a, m).mean(axis=0)

    return optimal_gaps, alpha_optim, alpha_list_agent, alpha_mean_list

if __name__ == "__main__":
    with open('first_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    beta = 10
    n_epochs = 20000
    sigma = 0.5
    step_size = 0.002

    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    X_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    Knm = compute_kernel_matrix(x_n, X_selected)
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)

    #W = np.ones((a, a))
    #W = W_base(a)
    W = fully_connected_graph(a)
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

    opt_gaps, alpha_optim, alpha_list, alpha_mean_list = gradient_tracking(
        x_n, y_n, X_selected, A, nu, sigma2, alpha_optimal, W, step_size, n_epochs=n_epochs)
    end = time.time()
    print(f'alpha optimal with Gradient Tracking : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with Gradient Tracking : {end - start}')
    
    agent_1 = np.linalg.norm(np.array(
        [alpha_list[i][0] for i in range(len(alpha_list))]) - alpha_optimal, axis=1)
    agent_2 = np.linalg.norm(np.array(
        [alpha_list[i][1] for i in range(len(alpha_list))]) - alpha_optimal, axis=1)
    agent_3 = np.linalg.norm(np.array(
        [alpha_list[i][2] for i in range(len(alpha_list))]) - alpha_optimal, axis=1)
    agent_4 = np.linalg.norm(np.array(
        [alpha_list[i][3] for i in range(len(alpha_list))]) - alpha_optimal, axis=1)
    agent_5 = np.linalg.norm(np.array(
        [alpha_list[i][4] for i in range(len(alpha_list))]) - alpha_optimal, axis=1)

    plt.plot(agent_1, label='Agent 1', color='blue')
    plt.plot(agent_2, label='Agent 2', color='red')
    plt.plot(agent_3, label='Agent 3', color='green')
    plt.plot(agent_4, label='Agent 4', color='orange')
    plt.plot(agent_5, label='Agent 5', color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality gap (norm)')
    plt.xscale("log")
    plt.yscale("log")
    #plt.savefig('opt_gaps_Gradient Tracking_with_agents_scalelog.png', bbox_inches='tight')
    plt.grid()
    plt.show()
    
 