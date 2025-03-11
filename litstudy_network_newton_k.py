import numpy as np
import pickle
import utils
from utils import compute_kernel_matrix, compute_alpha_star, compute_local_gradient

def B_block(W, i, j):
    m = W.shape[0]
    if i == j:
        B_ij = (1 - W[i, i]) * np.identity(m)
    else: 
        B_ij = W[i, j] * np.identity(m)
    return B_ij

def D_block( hessian_approx, sigma2, W, alpha, K_mm, i):
    D_ii = alpha * hessian_approx + 2 * B_block(W, i, i)
    return D_ii

def compute_local_gradient2(alpha, sigma2, K_mm, y_loc, K_im, nu, a, i):
    """
    Compute the local gradient.
    """
    
    
    return grad

def NN_K(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, K=1):
    """
    Network Newton-K (NN-K) distributed optimization algorithm.
    
    Parameters:
    - X, Y: Dataset and labels
    - X_selected: Subset of selected points
    - a: Number of agents
    - nu, sigma2: Optimization parameters
    - n_epochs: Number of iterations
    - alpha_star: Optimal solution (for comparison)
    - W: Weight matrix for communication
    - step_size: Learning rate
    - K: Number of Taylor series terms for Newton approximation
    
    Returns:
    - opt_gaps: List of optimality gaps per agent per iteration
    - alphas: List of alpha values per iteration
    """
    m = X_selected.shape[0]
    n = X.shape[0]
    a_data_idx = np.array_split(np.random.permutation(n), a)
    W = np.kron(W, np.ones((m, m)))
    alpha = np.ones((m * a, 1))
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    opt_gaps = [[] for _ in range(a)]
    alphas = []

    for epoch in range(n_epochs):
        print(epoch)
        alphas.append(alpha)
        gradients = np.zeros((m * a, 1))
        hessian_approx = np.zeros((m * a, m * a))

        for a_idx, data_idx in enumerate(a_data_idx):
            X_loc = X[data_idx]
            y_loc = Y[data_idx].reshape(-1, 1)
            Kim = compute_kernel_matrix(X_loc, X_selected)
            local_gradient = compute_local_gradient2(alpha[m * a_idx: m * (a_idx + 1)], sigma2, Kmm, y_loc, Kim, nu, a)
            gradients[m * a_idx: m * (a_idx + 1)] = local_gradient
            
            # Approximate Hessian (diagonal dominance)
            local_hessian = np.identity(m) + sigma2 * Kmm
            hessian_approx[m * a_idx: m * (a_idx + 1), m * a_idx: m * (a_idx + 1)] = np.linalg.inv(local_hessian)
        
        # Compute NN-K update (approximate Newton step)
        for _ in range(K):
            gradients = hessian_approx @ gradients
        
        alpha = W @ alpha - step_size * gradients
        
        for a_idx in range(a):
            alpha_agent = alpha[m * a_idx: m * (a_idx + 1)].reshape(-1, 1)
            opt_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            opt_gaps[a_idx].append(opt_gap)
    
    return opt_gaps, alphas


if "name" == "__main__":

    with open('first_database.pkl', 'rb') as f: # Load data
        x,y = pickle.load(f)
    
    # Parameters
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    
    # Generate data
    x_n=x[:n] 
    y_n=y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)

    # Set parameters
    K = 1
    max_iter = 100
    step_size = 0.1
    W = np
    n_epochs = 100
    alpha_star = compute_alpha_star(Kmm, Knm, y, sigma2, nu)
    
    # Call the function
    opt_gaps, alphas = NN_K(x_n, y_n, x_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, K)
    print(opt_gaps, alphas)