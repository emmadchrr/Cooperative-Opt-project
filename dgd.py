import numpy as np
from utils import *


def DGD(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size):
    """
    Decentralized gradient descent algorithm with multiple agents.
    """
    m = X_selected.shape[0]
    n = X.shape[0]

    a_data_idx = np.array_split(np.random.permutation(n), a)
    W = np.kron(W, np.ones((m, m)))
    alpha = np.ones((m*a, 1))
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    opt_gaps = [[] for _ in range(a)]
    alphas = []

    for epoch in range(n_epochs):
        alphas.append(alpha)
        gradients = np.zeros((m*a, 1))

        for a_idx, data_idx in enumerate(a_data_idx):
            X_loc = X[data_idx]
            y_loc = Y[data_idx].reshape(-1, 1)
            Kim = compute_kernel_matrix(X_loc, X_selected)
            local_gradient = compute_local_gradient(alpha[m * a_idx: m * (a_idx + 1)], sigma2, Kmm, y_loc, Kim, nu, a)
            gradients[m * a_idx: m * (a_idx + 1)] = local_gradient
        
        alpha = W @ alpha - step_size * gradients
        
        for a_idx in range(a):
            alpha_agent = alpha[m * a_idx: m * (a_idx + 1)].reshape(-1, 1)
            opt_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            opt_gaps[a_idx].append(opt_gap)
        
     
        
    return opt_gaps, alphas