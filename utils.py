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