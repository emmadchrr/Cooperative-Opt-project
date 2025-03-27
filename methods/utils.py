import numpy as np
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from tqdm import tqdm
import time
from sinkhorn_knopp import sinkhorn_knopp as skp


def euclidean_kernel(x, y):
    """
    Compute the Euclidean kernel between two vectors.
    """
    return np.exp(-np.linalg.norm(x - y)**2)


def compute_alpha_star(Kmm, Knm, y, sigma2, nu):
    """
    Compute the alpha_star vector.
    """
    m = Kmm.shape[0]
    A = sigma2 * Kmm + Knm.T @ Knm + nu * np.eye(m)
    b = Knm.T @ y
    
    return np.linalg.solve(A, b)

def local_objective_function(alpha, sigma2, K_mm, y_loc, K_im, nu, a):
    """
    Compute the local objective function value.
    """
    t1 = (sigma2 / (2 * a)) * alpha.T @ K_mm @ alpha
    t2 = 0.5 * np.sum((y_loc - K_im @ alpha) ** 2)
    t3 = (nu / (2 * a)) * np.linalg.norm(alpha) ** 2
    
    return t1 + t2 + t3

def nystrom_approx(alpha, X_selected, X):
    """
    Compute the Nystrom approximation.
    """
    K1m = compute_kernel_matrix(X, X_selected)
    return K1m @ alpha

def grad_alpha(sigma2, nu, Y, X, X_selected, alpha, a, m):
    """
    Calcule le gradient local pour chaque agent.
    """
    Kmm = compute_kernel_matrix(X_selected, X_selected)
    grad = np.zeros((a, m))
    
    if alpha.shape[0] != a * m:
        raise ValueError(f"Taille de alpha incorrecte: {alpha.shape}, attendu {(a * m, 1)}")
    
    alpha = alpha.reshape(a, m)  # Assurer une bonne indexation
    
    for i in range(a):
        K_im = compute_kernel_matrix(X[i].reshape(1, -1), X_selected)  # Assurer la bonne forme
        K_im_T = K_im.T
        Y_i = Y[i].reshape(-1, 1) if len(Y[i].shape) == 1 else Y[i]  # Assurer que Y_i a la bonne forme
        grad[i] = (sigma2 / a) * (Kmm @ alpha[i]) + K_im_T @ (K_im @ alpha[i] - Y_i) + (nu / a) * alpha[i]
    
    return grad

def compute_local_gradient(alpha, sigma2, K_mm, y_loc, K_im, nu, a):
    """
    Compute the local gradient.
    """
    grad = (sigma2 / a) * (K_mm @ alpha)
    grad += K_im.T @ (K_im @ alpha - y_loc)
    grad += (nu / a) * alpha
    
    return grad

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

def compute_local_Hessian1(sigma2, Kmm, Kim, nu, a):
    """ Compute local Hessian. """
    return (sigma2 / a) * Kmm + Kim.T @ Kim + (nu / a) * np.eye(Kmm.shape[0])

def compute_local_Hessian(sigma2, Kmm, Kmi, nu, a):
    """Compute local Hessian for an agent"""
    return Kmm + (sigma2/(nu*a)) * Kmi @ Kmi.T

def W_base(a):
    W = np.identity(a)
    for i in range(4):
        W[i, i+1]=1/3
        W[i+1, i]=1/3
    W[0, a-1]=1/3
    W[a-1,0]=1/3
    return W

def W_base_bis(a):
    W = np.identity(a)
    for i in range(4):
        W[i, i+1]=1
        W[i+1, i]=1
    W[0, a-1]=1
    W[a-1,0]=1
    return W

def normalize_adjacency_matrix(adj_matrix):
    """
    Normalize a given adjacency matrix by ensuring row and column sums are balanced.
    """
    row_totals = np.sum(adj_matrix, axis=1, keepdims=True)
    adj_matrix /= row_totals  # Normalize rows
    col_totals = np.sum(adj_matrix, axis=0, keepdims=True)
    adj_matrix /= col_totals  # Normalize columns
    return adj_matrix

def fully_connected_graph(num_nodes):
    """
    Constructs a fully connected graph where all nodes are interconnected.
    """
    return normalize_adjacency_matrix(np.ones((num_nodes, num_nodes)))

def linear_graph(num_nodes):
    """
    Generates a linear graph where each node (except endpoints) is connected to two neighbors.
    """
    adj_matrix = np.eye(num_nodes)
    for i in range(num_nodes - 1):
        adj_matrix[i, i + 1] = 1
        adj_matrix[i + 1, i] = 1
    return normalize_adjacency_matrix(adj_matrix)

def small_world_graph(num_nodes, rewiring_prob=0.1):
    """
    Generates a small-world network using the Watts-Strogatz model.
    """
    small_world_net = nx.watts_strogatz_graph(num_nodes, k=2, p=rewiring_prob)
    adj_matrix = nx.to_numpy_array(small_world_net)
    np.fill_diagonal(adj_matrix, 1)
    return normalize_adjacency_matrix(adj_matrix)

def compute_alpha(x, y, x_selected, sigma, mu):
    n = len(x)
    m = len(x_selected)
    Kmm = compute_kernel_matrix(x_selected, x_selected)
    Knm = compute_kernel_matrix(x[0:n], x_selected)
    alpha_exact = np.linalg.inv(
        sigma**2*Kmm + mu*np.eye(m) + np.transpose(Knm) @ Knm) @ np.transpose(Knm) @ y[0:n]
    return alpha_exact