from utils import *
import numpy as np

    
def solve_alpha_dualdec(x, y, selected_points, selected_points_agent, sigma, nu, adj_matrix, lamb):
    # print("lamb shape : ", lamb.shape)
    n = len(x)
    a = len(selected_points_agent)
    m = len(selected_points)
    Kmm = compute_kernel_matrix(x[selected_points], x[selected_points])
    alpha = []
    for i in range(a):
        Kim = compute_kernel_matrix(x[selected_points_agent[i]], x[selected_points])
        A = sigma**2 * Kmm + np.eye(m)*nu + np.transpose(Kim) @ Kim
        b = np.transpose(Kim) @ y[selected_points_agent[i]]
        for j in range(a):
            if adj_matrix[i, j] != 0:
                if i > j:
                    b-= lamb[i, j, :]
                else:
                    b+= lamb[j, i, :]
        alpha.append(np.linalg.solve(A, b))
    return np.array(alpha)

def dualDec(x, y, selected_points, selected_points_agent, K, sigma, nu, lr, W, max_iter=1000, lamb0=0):

    graph = 1 * (W>0)
    m = len(selected_points)
    a = len(selected_points_agent)
    for i in range(a):
        graph[i, i] = 0
    lambda_ij = lamb0*np.ones((a, a, m)) # should be shape number of edges in comnunication graph
    alpha_mean_list = []
    alpha_list_agent = []
    for _ in tqdm(range(max_iter)):
        alpha_optim = np.zeros((a,m))
        alpha_optim = solve_alpha_dualdec(
            x, y, selected_points, selected_points_agent, sigma, nu,
            K, graph, lambda_ij)
        for i in range(a):
            for j in range(i):
                lambda_ij[i, j, : ] += lr * (alpha_optim[i, :] - alpha_optim[j, :])
        alpha_mean_list.append(alpha_optim.mean(axis=0))
        alpha_list_agent.append(alpha_optim)
    
    alpha_optim = alpha_optim.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)

    return alpha_optim, alpha_list_agent, alpha_mean_list