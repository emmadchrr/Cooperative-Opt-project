import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
from utils import *

# Global Constants
B = 20
C = 5
E = 5
learning_rate = 0.002
num_SAMPLES = 100
num_FEATURES = 10
REGULARIZATION = 5
sigma = 0.5
sigma2 = 0.25
nu = 1.0

def compute_kernel_fedavg(indices, selected_x, x):
    """Compute the kernel matrix between selected points and all data points."""
    print(f"Computing kernel with {len(indices)} indices and {len(selected_x)} selected points")
    return np.array([[np.exp(-(x[j] - x_i) ** 2) for x_i in selected_x] for j in indices])

def compute_gradient(alpha, y, Knm, Kmm, client_idx, batch_dict):
    """Compute the gradient for local model updates."""
    grad = (1 / REGULARIZATION) * sigma2 * Kmm.dot(alpha) 
    grad += nu * (1 / REGULARIZATION) * alpha 
    grad += sum([Knm[j] * (Knm[j].dot(alpha)) - y[j] * Knm[j] for j in batch_dict[client_idx]])
    return grad

def train_federated(E, optimal_alpha, Knm, Kmm, y, batch_dict, num_clients, max_iter=10000):
    """Run the federated training pipeline."""
    x = np.random.randn(num_FEATURES) * 0.01
    error_history = [np.linalg.norm(x - optimal_alpha)]
    print(f"Initial error: {error_history[0]}")

    for iter in range(max_iter // E):
        local_models = local_training_step(x, Knm, Kmm, y, batch_dict, num_clients, E)
        x = sum(local_models) / len(local_models)
        current_error = np.linalg.norm(x - optimal_alpha)
        error_history.append(current_error)
        if iter % 100 == 0:
            print(f"Iteration {iter*E}: error = {current_error}")
    
    return error_history


def local_training_step(x, Knm, Kmm, y, batch_dict, num_clients, epochs):
    """Perform local training updates on client models."""
    local_models = [x.copy() for _ in range(num_clients)]
    for j in range(num_clients):
        for _ in range(epochs):
            # Ensure we only use valid batch indices
            valid_batch = [idx for idx in batch_dict[j] if idx < len(y)]
            if not valid_batch:
                continue
                
            grad = (1 / REGULARIZATION) * sigma2 * Kmm.dot(local_models[j]) 
            grad += nu * (1 / REGULARIZATION) * local_models[j]
            grad += sum([Knm[idx] * (Knm[idx].dot(local_models[j])) - y[idx] * Knm[idx] for idx in valid_batch])
            local_models[j] -= learning_rate * grad
    return local_models

def plot_results():
    """Load data and plot results for different epoch values."""
    print("Starting plot_results")
    
    try:
        with open("second_database.pkl", "rb") as f:
            x, y = pickle.load(f)
            print("Successfully loaded data")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    x = np.array(x)
    y = np.array(y)
    print(f"Data shapes - x: {x.shape}, y: {y.shape}")

    # Flatten if 2D
    if len(x.shape) == 2:
        print("Flattening 2D arrays")
        x = x.flatten()
        y = y.flatten()

    actual_num_samples = len(x)
    print(f"Actual number of samples: {actual_num_samples}")

    # Adjust constants if needed
    global num_SAMPLES, num_FEATURES
    num_SAMPLES = min(num_SAMPLES, actual_num_samples)
    num_FEATURES = min(num_FEATURES, actual_num_samples)
    print(f"Using num_SAMPLES={num_SAMPLES}, num_FEATURES={num_FEATURES}")

    plt.figure(figsize=(10, 6))
    
    # Create data points
    sel = np.arange(actual_num_samples)
    np.random.seed(0)
    data_points = np.random.choice(sel, num_SAMPLES, replace=False)
    
    # Select feature points
    indices = np.random.choice(len(data_points), num_FEATURES, replace=False)
    selected_x = x[data_points[indices]]
    y_n = y[data_points]
    
    print(f"Selected {len(data_points)} data points and {len(indices)} features")

    # Initialize client batches
    B_algo = {}
    for i in range(C):
        start = i * B
        end = min((i + 1) * B, len(data_points))
        if start >= end:
            print(f"Warning: Not enough data for client {i} (start={start}, end={end})")
            continue
        B_algo[i] = data_points[start:end]
    
    print(f"Created batches for {len(B_algo)} clients")

    # Compute kernels
    Knm = compute_kernel_fedavg(data_points, selected_x, x)
    Kmm = compute_kernel_fedavg(indices, selected_x, x)
    print(f"Kernel shapes - Knm: {Knm.shape}, Kmm: {Kmm.shape}")

    # Compute optimal alpha
    try:
        term1 = nu * np.identity(num_FEATURES)
        term2 = sigma * sigma * Kmm
        term3 = Knm.T.dot(Knm)
        inverse_term = np.linalg.inv(term1 + term2 + term3)
        alpha_star = inverse_term.dot(Knm.T.dot(y_n))
        print("Computed alpha_star successfully")
    except Exception as e:
        print(f"Error computing alpha_star: {e}")
        return

    colors = {1: "royalblue", 5: "orange", 50: "forestgreen"}
    
    for E_val in [1, 5, 50]:
        print(f"\nTraining with E = {E_val}")
        
        try:
            error_history = train_federated(E_val, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=10000)
            
            x_plot = [k * E_val for k in range(len(error_history))]
            plt.plot(x_plot, error_history, label=f"E = {E_val}", color=colors[E_val])
            print(f"Completed training for E = {E_val}")
            
        except Exception as e:
            print(f"Error during training for E = {E_val}: {e}")
            continue

    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.title("Federated Learning with Different Epoch Values")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    print("Plotting complete")


def plot_results_different_C():
    """Load data and plot results for different client numbers."""
    print("Starting plot_results_different_C")
    
    try:
        with open("first_database.pkl", "rb") as f:
            x, y = pickle.load(f)
            print("Successfully loaded data")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    x = np.array(x)
    y = np.array(y)
    print(f"Data shapes - x: {x.shape}, y: {y.shape}")

    actual_num_samples = len(x)
    print(f"Actual number of samples: {actual_num_samples}")

    # Adjust constants if needed
    global num_SAMPLES, num_FEATURES
    num_SAMPLES = min(num_SAMPLES, actual_num_samples)
    num_FEATURES = min(num_FEATURES, 20)  # Limit features to prevent overfitting
    print(f"Using num_SAMPLES={num_SAMPLES}, num_FEATURES={num_FEATURES}")

    plt.figure(figsize=(10, 6))
    colors = {5: "royalblue", 10: "orange", 15: "pink", 20: "forestgreen"}
    has_valid_plots = False

    for C_val in [5, 10, 15, 20]:
        print(f"\nProcessing C = {C_val}")
        
        try:
            # Calculate how many samples we can use (B samples per client)
            total_samples_needed = C_val * B
            actual_samples_to_use = min(total_samples_needed, actual_num_samples)
            
            # Select random data points (indices)
            np.random.seed(0)
            data_indices = np.random.choice(actual_num_samples, actual_samples_to_use, replace=False)
            x_subset = x[data_indices]
            y_subset = y[data_indices]
            
            print(f"Selected {len(data_indices)} data points")

            # Select feature points from the subset
            feature_indices = np.random.choice(len(data_indices), num_FEATURES, replace=False)
            selected_x = x_subset[feature_indices]
            
            # Initialize client batches (using indices relative to the subset)
            B_algo = {}
            actual_clients = min(C_val, actual_samples_to_use // B)
            for i in range(actual_clients):
                start = i * B
                end = (i + 1) * B
                B_algo[i] = np.arange(start, end)  # Using relative indices
                
            print(f"Created batches for {actual_clients} clients")

            # Compute kernels using only the subset data
            Knm = compute_kernel_fedavg(np.arange(len(x_subset)), selected_x, x_subset)
            Kmm = compute_kernel_fedavg(feature_indices, selected_x, x_subset)
            print(f"Kernel shapes - Knm: {Knm.shape}, Kmm: {Kmm.shape}")

            # Compute optimal alpha using subset data
            try:
                term1 = nu * np.identity(num_FEATURES)
                term2 = sigma * sigma * Kmm
                term3 = Knm.T.dot(Knm)
                inverse_term = np.linalg.inv(term1 + term2 + term3)
                alpha_star = inverse_term.dot(Knm.T.dot(y_subset))
                print(f"Computed alpha_star with shape {alpha_star.shape}")
            except Exception as e:
                print(f"Error computing alpha_star: {e}")
                continue

            # Train federated model using subset data
            print("Starting federated training")
            error_history = train_federated(E, alpha_star, Knm, Kmm, y_subset, B_algo, actual_clients, max_iter=10000)
            
            # Plot results
            x_plot = [k * E for k in range(len(error_history))]
            plt.plot(x_plot, error_history, label=f"C = {C_val}", color=colors[C_val])
            has_valid_plots = True
            
        except Exception as e:
            print(f"Error processing C = {C_val}: {e}")
            continue

    if not has_valid_plots:
        print("Warning: No valid plots were generated")
        plt.close()
        return

    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.title("Federated Learning with Different Client Numbers")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    print("Plotting complete")

def plot_results_different_sel_clients():
    """Load data and plot results for different epoch values."""
    print("Starting plot_results_different_sel_clients")
    
    try:
        with open("second_database.pkl", "rb") as f:
            x, y = pickle.load(f)
            print("Successfully loaded data")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    x = np.array(x)
    y = np.array(y)
    print(f"Data shapes - x: {x.shape}, y: {y.shape}")

    # Flatten if 2D
    if len(x.shape) == 2:
        print("Flattening 2D arrays")
        x = x.flatten()
        y = y.flatten()

    actual_num_samples = len(x)
    print(f"Actual number of samples: {actual_num_samples}")

    # Adjust constants if needed
    global num_SAMPLES, num_FEATURES
    num_SAMPLES = min(num_SAMPLES, actual_num_samples)
    num_FEATURES = min(num_FEATURES, actual_num_samples)
    print(f"Using num_SAMPLES={num_SAMPLES}, num_FEATURES={num_FEATURES}")

    plt.figure(figsize=(10, 6))
    
    for nb in [0, 1, 2, 3]:
        print(f"\nProcessing random selection {nb}")
        
        try:
            # Create data points
            sel = np.arange(actual_num_samples)
            np.random.seed(nb)
            data_points = np.random.choice(sel, num_SAMPLES, replace=False)
            
            # Select feature points
            indices = np.random.choice(len(data_points), num_FEATURES, replace=False)
            selected_x = x[data_points[indices]]
            y_n = y[data_points]
            
            print(f"Selected {len(data_points)} data points and {len(indices)} features")

            # Initialize client batches
            B_algo = {}
            for i in range(C):
                start = i * B
                end = min((i + 1) * B, len(data_points))
                if start >= end:
                    print(f"Warning: Not enough data for client {i} (start={start}, end={end})")
                    continue
                B_algo[i] = data_points[start:end]
            
            print(f"Created batches for {len(B_algo)} clients")

            # Compute kernels
            Knm = compute_kernel_fedavg(data_points, selected_x, x)
            Kmm = compute_kernel_fedavg(indices, selected_x, x)
            print(f"Kernel shapes - Knm: {Knm.shape}, Kmm: {Kmm.shape}")

            # Compute optimal alpha
            try:
                term1 = nu * np.identity(num_FEATURES)
                term2 = sigma * sigma * Kmm
                term3 = Knm.T.dot(Knm)
                inverse_term = np.linalg.inv(term1 + term2 + term3)
                alpha_star = inverse_term.dot(Knm.T.dot(y_n))
                print("Computed alpha_star successfully")
            except Exception as e:
                print(f"Error computing alpha_star: {e}")
                continue

            # Train federated model
            error_history = train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=10000)
            
            # Plot results
            x_plot = [k * E for k in range(len(error_history))]
            colors = {0: "royalblue", 1: "orange", 2: "pink", 3: "forestgreen"}
            plt.plot(x_plot, error_history, label=f"selection {nb}", color=colors[nb])
            
        except Exception as e:
            print(f"Error processing selection {nb}: {e}")
            continue

    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.title("Federated Learning with Different Random Selections")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    print("Plotting complete")

if __name__ == "__main__":
    print("Starting main execution")
    #plot_results_different_sel_clients()
    #plot_results_different_C()
    plot_results()
    print("Program finished")