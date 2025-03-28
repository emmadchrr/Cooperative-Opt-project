import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *

def grad_alpha_fedavg(sigma, nu, y, alpha, Kmm, Kim, C, m):
    grad = np.zeros((C, m))
    for i in range(C):
        big_kernel_im = Kim[i]
        grad[i] = (1/C) * (sigma**2 * Kmm + nu * np.eye(m)) @ alpha[i] + \
                 big_kernel_im.T @ (big_kernel_im @ alpha[i] - y[i])
    return grad

def fedAvg(X, Y, x_m_points, T, E, K, Kim, sigma, nu, lr, B=None):
    """
    Args:
        B: Batch size (if None, uses full client data)
        C: Number of clients (determined from X)
        E: Number of local epochs
    """
    m = len(x_m_points)
    C = len(X)
    alpha_server = np.zeros(m)
    alpha_agents = np.zeros((C, m))
    optimal_gaps = []
    
    # Compute true optimal solution using all data
    alpha_optim = compute_alpha(np.concatenate(X), np.concatenate(Y), x_m_points, sigma, nu)
    total_iterations = 0
    
    for t in range(T):
        # Client updates
        for epoch in range(E):
            total_iterations += 1
            if total_iterations > 5000:  # Critère d'arrêt
                print(f"Stopping early at T={t+1}, Epoch={epoch+1}, Total Iterations={total_iterations}")
                return alpha_server, optimal_gaps
            grad = grad_alpha_fedavg(sigma, nu, Y, alpha_agents, K, Kim, C, m)
            alpha_agents -= lr * grad
        
        # Server aggregation
        alpha_server = np.mean(alpha_agents, axis=0)
        current_gap = np.linalg.norm(alpha_server - alpha_optim)
        optimal_gaps.append(current_gap)
        
        # Broadcast updated model
        alpha_agents = np.tile(alpha_server, (C, 1))
        
        if t % 10 == 0:
            print(f"Round {t+1}/{T}, Gap: {current_gap:.4f}")

    return alpha_server, optimal_gaps

def run_experiments(X_full, Y_full, config):
    """Run all experiments with given configuration"""
    # Base parameters
    base_m = config['base_m']
    x_m_points = np.linspace(-1, 1, base_m)
    K = compute_kernel_matrix(x_m_points, x_m_points)
    
    # Experiment 1: Varying epochs (E)
    plt.figure(figsize=(12, 8))
    for E in config['E_values']:
        Kim = [compute_kernel_matrix(X_full[i], x_m_points) for i in range(config['C'])]
        _, gaps = fedAvg(X_full[:config['C']], Y_full[:config['C']], x_m_points, 
                        config['T'], E, K, Kim, config['sigma'], config['nu'], config['lr'])
        plt.plot([k*E for k in range(len(gaps))], gaps, label=f'E={E}')
    
    plt.xlabel('Total Iterations (T×E)')
    plt.ylabel('Optimal Gap')
    plt.title(f'Varying Epochs (C={config["C"]}, B={config["B"]})')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Experiment 2: Varying batch sizes (m)
    plt.figure(figsize=(12, 8))
    for m in config['m_values']:
        x_m = np.linspace(-1, 1, m)
        K_m = compute_kernel_matrix(x_m, x_m)
        Kim = [compute_kernel_matrix(X_full[i], x_m) for i in range(config['C'])]
        _, gaps = fedAvg(X_full[:config['C']], Y_full[:config['C']], x_m, 
                        config['T'], config['E'], K_m, Kim, config['sigma'], config['nu'], config['lr'])
        plt.plot([k*config['E'] for k in range(len(gaps))], gaps, label=f'B={m}')
    
    plt.xlabel('Total Iterations (T×E)')
    plt.ylabel('Optimal Gap')
    plt.title(f'Varying Batch Sizes (C={config["C"]}, E={config["E"]})')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Experiment 3: Varying client selections
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, config['num_selections']))
    for i in range(config['num_selections']):
        np.random.seed(i)
        selected = np.random.choice(len(X_full), config['C'], replace=False)
        X_sel = X_full[selected]
        Y_sel = Y_full[selected]
        Kim = [compute_kernel_matrix(X_sel[i], x_m_points) for i in range(config['C'])]
        _, gaps = fedAvg(X_sel, Y_sel, x_m_points, 
                        config['T'], config['E'], K, Kim, config['sigma'], config['nu'], config['lr'])
        plt.plot(range(len(gaps)), gaps, color=colors[i], 
               label=f'Selection {i+1}: Clients {selected}')
    
    plt.xlabel('Communication Rounds (T)')
    plt.ylabel('Optimal Gap')
    plt.title(f'Client Selection Variability (C={config["C"]}, E={config["E"]})')
    plt.yscale('log')
    plt.xscale
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data
    with open('second_database.pkl', 'rb') as f:
        X_full, Y_full = pickle.load(f)
    X_full = np.array(X_full)
    Y_full = np.array(Y_full)
    
    # Configuration
    config = {
        'C': 5,          # Number of clients
        'B': 20,         # Batch size
        'E': 20,          # Base number of epochs
        'T': 10000,         # Communication rounds
        'base_m': 10,    # Base batch size
        'sigma': 0.5,
        'nu': 10,
        'lr': 0.002,
        'E_values': [1, 5, 50],
        'm_values': [5, 10, 15, 20],
        'num_selections': 5
    }
    
    print(f"Running experiments with C={config['C']} clients, B={config['B']} batch size")
    run_experiments(X_full, Y_full, config)