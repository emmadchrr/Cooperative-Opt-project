from utils import *
import numpy as np

    
def solve_alpha_dualdec(x, y, selected_points, selected_points_agent, sigma, nu, adj_matrix, lamb):
    # print("lamb shape : ", lamb.shape)
    n = len(x)
    a = len(selected_points_agent)
    m = len(selected_points)
    Kmm = compute_kernel_matrix(selected_points, selected_points)
    alpha = []
    for i in range(a):
        Kim = compute_kernel_matrix(x[selected_points_agent[i]], selected_points)
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
            graph, lambda_ij)
        for i in range(a):
            for j in range(i):
                lambda_ij[i, j, : ] += lr * (alpha_optim[i, :] - alpha_optim[j, :])
        alpha_mean_list.append(alpha_optim.mean(axis=0))
        alpha_list_agent.append(alpha_optim)
    
    alpha_optim = alpha_optim.reshape(a, m)
    alpha_optim = np.mean(alpha_optim, axis=0)

    return alpha_optim, alpha_list_agent, alpha_mean_list

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
    W = np.ones((a,a))
    K = compute_kernel_matrix(x_n, x_n)
    selected_pts_agents = np.array_split(np.random.permutation(n), a)
    lr = 0.01

    start = time.time()
    alpha_optimal = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    end = time.time()
    print(f'Time to compute alpha optimal : {end - start}\n')
    print(f'Optimal alpha : {alpha_optimal}\n')
    start = time.time()
    alpha_optim, alpha_list, alpha_mean_list = dualDec(
        x_n, y_n, x_selected, selected_pts_agents,
        K, sigma, nu, 0.01, W, max_iter=1000, lamb0=0.
    )
    end = time.time()
    print(f'alpha optimal with dual decomposition : {alpha_optim}')
    print(
        f'Time to compute alpha optimal with dual decomposition : {end - start}')
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
    plt.grid()
    plt.show()
    # Plot selected points and the prediction of the model with the alpha optimal 
    plt.figure(0)
    for i in range(a):
        plt.plot(x_n[i], y_n[i], 'o', label=f'Agent {i+1}')
    # plt.plot(x[0:n], y[0:n], 'o', label='Data')
    x_predict = np.linspace(-1, 1, 250)
    K_f = compute_kernel_matrix(x_predict, x_selected)
    # fx_predict = get_Kij(range(n), selected_points, K) @ alpha_optim_gt
    fx_predict = K_f @ alpha_optim
    plt.plot(x_predict, fx_predict, label='Prediction')
    plt.grid()
    plt.legend()
    plt.show()