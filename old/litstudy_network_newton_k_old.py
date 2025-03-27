import numpy as np
from dgd import DGD
import pickle
import matplotlib.pyplot as plt
import time
from methods.utils import *

def compute_local_Hessian(sigma2, Kmm, Kim, nu, a):
    """ Compute local Hessian. """
    return (sigma2 / a) * Kmm + Kim.T @ Kim + (nu / a) * np.eye(Kmm.shape[0])


def network_newton_k(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, K):
    """
    Implements the Network Newton-K method.

    """
    
    m = X_selected.shape[0]  # Number of selected points per agent
    n = X.shape[0]  # Total number of data points
    alpha = np.zeros((m * a, 1))  # Initialize alpha
    opt_gaps = [[] for _ in range(a)]  # Store optimality gaps

    # Split data across agents
    a_data_idx = np.array_split(np.random.permutation(n), a)

    # Compute Kernel Matrices
    Kmm = compute_kernel_matrix(X_selected, X_selected)

    for epoch in range(n_epochs):
        gradients = np.zeros((m * a, 1))
        Hessians = np.zeros((m * a, m * a))

        # Compute Local Gradients and Hessians
        for i in range(a):
            data_idx = a_data_idx[i]
            X_loc = X[data_idx]
            y_loc = Y[data_idx].reshape(-1, 1)
            Kim = compute_kernel_matrix(X_loc, X_selected)

            local_grad = compute_local_gradient(alpha[m*i:m*(i+1)], sigma2, Kmm, y_loc, Kim, nu, a)
            gradients[m*i:m*(i+1)] = local_grad

            local_Hessian = compute_local_Hessian(sigma2, Kmm, Kim, nu, a)
            Hessians[m*i:m*(i+1), m*i:m*(i+1)] = local_Hessian

        gradients /= np.linalg.norm(gradients) + 1e-6  
        # Compute D and B Matrices
        D = np.diagflat(np.diag(Hessians))  # Extract diagonal as full matrix
        B = Hessians - D  # Compute off-diagonal part
        D_inv = np.linalg.inv(D + 1e-2 * np.eye(D.shape[0]))  # Regularized inverse
        B_D_inv = B @ D_inv

        # Ensure Convergence of Truncated Taylor Series
        rho = np.linalg.norm(B_D_inv, ord=2)  # Compute spectral norm

        if rho >= 1:
            print(f"Reducing Newton steps because ρ={rho:.3f} ≥ 1")
            K = max(1, K // 2)  # Reduce K dynamically


        # Truncated Taylor Approximation of H⁻¹
        H_inv_approx = D_inv.copy()
        B_D_inv_power = B_D_inv.copy()
        for k in range(1, K+1):
            H_inv_approx += (-1)**k * B_D_inv_power @ D_inv
            B_D_inv_power = B_D_inv_power @ B_D_inv  # Update power

        # Newton Step Update
        alpha = alpha - step_size * H_inv_approx @ gradients

        # Apply Network Consensus Step
        alpha = (W @ alpha.reshape(a, m)).reshape(m * a, 1)

        # Compute Optimality Gaps
        for i in range(a):
            alpha_agent = alpha[m*i:m*(i+1)].reshape(-1, 1)
            opt_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            opt_gaps[i].append(opt_gap)

    return opt_gaps, alpha

def run_comparison(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, Ks):
    from copy import deepcopy
    from tqdm import tqdm

    def nn(K):
        """Implementation of Network Newton with varying K"""
        optimal_gaps, _ = network_newton_k(deepcopy(X), deepcopy(Y), X_selected, a, nu, sigma2, 
                                     n_epochs, alpha_star, W, step_size, K)
        return optimal_gaps

    def dgd():
        """Implementation of DGD"""
        opt_gaps, _ , _ , _ = DGD(deepcopy(X), deepcopy(Y), X_selected, a, nu, sigma2, alpha_star, W, step_size, n_epochs=500)
        return opt_gaps

    # Run DGD as a baseline
    opt_gaps_dgd = dgd()

    # Run Network Newton for different values of K
    optimal_gaps_nn = {K: nn(K) for K in Ks}

    # Plot results
    plt.figure(figsize=(12, 7))

    # Plot DGD
    for i in range(a):
        plt.plot(opt_gaps_dgd[i], label=f'DGD - Agent {i+1}', linestyle='solid', alpha=0.5)

    # Plot Network Newton for different K values
    for K in Ks:
        for i in range(a):
            plt.plot(optimal_gaps_nn[K][i], label=f'NN-K={K} - Agent {i+1}', linestyle='dashed', alpha=0.5)

    plt.xlabel('Epochs')
    plt.ylabel('Optimality Gap')
    plt.title('Comparison of Network Newton (varied K) vs. DGD')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log')  # Échelle logarithmique pour X
    plt.yscale('log')  # Échelle logarithmique pour Y (optionnel)
    plt.grid(True)
    #plt.savefig('optimality_gaps_with_agents_scalelog.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Example usage of Network_Newton
    with open('first_database.pkl', 'rb') as f: # Load data
        x,y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    Ks = [1, 3, 5]  # Different values of K to test
    # Generate data
    x_n = x[:n] 
    y_n = y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    n_epochs = 100
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu)
    W = np.zeros((a,a))
    step_size = 0.01


    run_comparison(x_n, y_n, x_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, Ks)



    # Initialize results

    #results = {}

    # Run DGD
    #start_time = time.time()
    #opt_gaps_dgd, alphas_dgd = DGD(x_n, y_n, x_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size)
    #dgd_time = time.time() - start_time
    #results['DGD'] = (opt_gaps_dgd, dgd_time, alphas_dgd)

    # Run Network Newton for different K values
    #for K in Ks:
    #    start_time = time.time()
    #    opt_gaps_nn, alphas_nn = Network_Newton(x_n, y_n, x_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, K)
    #    nn_time = time.time() - start_time
    #    results[f'NN-K={K}'] = (opt_gaps_nn, nn_time, alphas_nn)

    # Plot optimality gaps
    #plt.figure(figsize=(10, 6))
    #for key, (opt_gaps, _, _) in results.items():
    #    plt.plot(np.mean(opt_gaps, axis=0), label=key)
    #plt.xlabel('Iterations')
    #plt.ylabel('Optimality Gap')
    #plt.title('Optimality Gap Comparison of DGD and NN Methods')
    #plt.legend()
    #plt.grid(True)
    #plt.savefig('optimality_gaps.png')
    #plt.show()

    # Print convergence times
    #for key, (_, runtime, _) in results.items():
    #    print(f'{key} Convergence Time: {runtime:.4f} seconds')
