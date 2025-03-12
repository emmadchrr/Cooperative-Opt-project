import numpy as np
import pickle
import matplotlib.pyplot as plt

def compute_alpha_star(Kmm, Knm, y, sigma2, nu):
    """
    Compute the alpha_star vector.
    """
    m = Kmm.shape[0]
    A = sigma2 * Kmm + Knm.T @ Knm + nu * np.eye(m)
    b = Knm.T @ y
    
    return np.linalg.solve(A, b)

def compute_local_Hessian(sigma2, Kmm, Kim, nu, a):
    """
    Compute the local Hessian matrix.
    """
    Hessian = (sigma2 / a) * Kmm
    Hessian += Kim.T @ Kim
    Hessian += (nu / a) * np.eye(Kmm.shape[0])
    
    return Hessian

def compute_kernel_matrix(X, Y):
    """
    Compute the kernel matrix between two sets of vectors.
    """
    rows, cols = X.shape[0], Y.shape[0]
    K = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            K[i, j] = euclidean_kernel(X[i], Y[j])
    
    return K

def euclidean_kernel(x, y):
    """
    Euclidean kernel function.
    """
    return np.exp(-np.linalg.norm(x - y)**2)

def compute_local_gradient(alpha, sigma2, K_mm, y_loc, K_im, nu, a):
    """
    Compute the local gradient.
    """
    grad = (sigma2 / a) * (K_mm @ alpha)
    grad += K_im.T @ (K_im @ alpha - y_loc)
    grad += (nu / a) * alpha
    
    return grad


def Network_Newton(X, Y, X_selected, a, nu, sigma2, n_epochs, alpha_star, W, step_size, K):
    """
    Network Newton algorithm for decentralized optimization.
    
    Parameters:
    - X: Input data matrix (n x d)
    - Y: Output vector (n x 1)
    - X_selected: Selected data points for kernel computation (m x d)
    - a: Number of agents
    - nu: Regularization parameter
    - sigma2: Kernel parameter
    - n_epochs: Number of epochs
    - alpha_star: Optimal alpha (for computing optimality gap)
    - W: Weight matrix (a x a)
    - step_size: Step size for gradient descent
    - K: Number of Taylor series terms to keep in the Newton step approximation
    
    Returns:
    - opt_gaps: List of optimality gaps for each agent
    - alphas: List of alpha values over epochs
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
        print(f"Epoch {epoch}")
        alphas.append(alpha)
        gradients = np.zeros((m*a, 1))
        Hessians = np.zeros((m*a, m*a))

        # Compute local gradients and Hessians
        for a_idx, data_idx in enumerate(a_data_idx):
            X_loc = X[data_idx]
            y_loc = Y[data_idx].reshape(-1, 1)
            Kim = compute_kernel_matrix(X_loc, X_selected)
            
            # Local gradient
            local_gradient = compute_local_gradient(alpha[m * a_idx: m * (a_idx + 1)], sigma2, Kmm, y_loc, Kim, nu, a)
            gradients[m * a_idx: m * (a_idx + 1)] = local_gradient
            
            # Local Hessian
            local_Hessian = compute_local_Hessian(sigma2, Kmm, Kim, nu, a)
            Hessians[m * a_idx: m * (a_idx + 1), m * a_idx: m * (a_idx + 1)] = local_Hessian
        
        # Approximate Newton step using truncated Taylor series
        D = np.diag(np.diag(Hessians))  # Diagonal part of Hessian
        B = Hessians - D  # Off-diagonal part of Hessian
        D_inv = np.linalg.pinv(D + 1e-6 * np.eye(D.shape[0]))  # Adds small regularization
        B_D_inv = B @ D_inv
        
        # Truncated Taylor series approximation of Hessian inverse
        H_inv_approx = D_inv
        for k in range(1, K+1):
            H_inv_approx += (-1)**k * np.linalg.matrix_power(B_D_inv, k) @ D_inv
        
        # Update alpha using the approximate Newton step
        alpha = W @ alpha - step_size * H_inv_approx @ gradients
        
        # Compute optimality gaps
        for a_idx in range(a):
            alpha_agent = alpha[m * a_idx: m * (a_idx + 1)].reshape(-1, 1)
            opt_gap = np.linalg.norm(alpha_agent - alpha_star.reshape(-1, 1))
            opt_gaps[a_idx].append(opt_gap)
        
    return opt_gaps, alphas

if __name__ == "__main__":
    # Example usage of Network_Newton
    with open('first_database.pkl', 'rb') as f: # Load data
        x,y = pickle.load(f)
    
    n, m, a = 100, 10, 5
    sigma2 = 0.25
    nu = 1
    K = 3
    # Generate data
    x_n=x[:n] 
    y_n=y[:n]

    sel = [i for i in range(n)]
    ind = np.random.choice(sel, m, replace=False)
    x_selected = np.array([x[i] for i in ind])
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x_n, x_selected)
    # Run Network_Newton
    opt_gaps, alphas = Network_Newton(X=x_n, Y=y_n, X_selected=x_selected, a=a, nu=nu, sigma2=sigma2, n_epochs=100, 
    alpha_star = compute_alpha_star(Kmm, Knm, y_n, sigma2, nu), W=np.ones((a, a)), step_size=0.01, K = K)
        
    # Print the results
    #print("Optimality gaps:", opt_gaps)
    #print("Alphas:", alphas)
    plt.figure(figsize=(10, 8))
    for i in range(a):
        plt.plot(opt_gaps[i], label=f'Agent {i+1}')
    plt.xlabel('Iterations')
    plt.ylabel('Optimality Gap')
    plt.title('Optimality Gaps for Each Agent')
    plt.legend()
    plt.grid(True)
    plt.show()
