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
    """
    Compute the kernel matrix between selected points and all data points.
    """
    return np.array([[np.exp(-(x[j] - x_i) ** 2) for x_i in selected_x] for j in indices])


def compute_gradient(alpha, y, Knm, Kmm, client_idx, batch_dict):
    """
    Compute the gradient for local model updates.
    """
    return (1 / REGULARIZATION) * sigma2 * Kmm.dot(alpha) + nu * (1 / REGULARIZATION) * alpha + sum([
        Knm[j] * (Knm[j].dot(alpha)) - y[j] * Knm[j] for j in batch_dict[client_idx]
    ])


def local_training_step(x, Knm, Kmm, y, batch_dict, num_clients, epochs):
    """
    Perform local training updates on client models.
    """
    local_models = [x.copy() for _ in range(num_clients)]
    for j in range(num_clients):
        for _ in range(epochs):
            local_models[j] -= learning_rate * compute_gradient(local_models[j], y, Knm, Kmm, j, batch_dict)
    return local_models

def train_federated(E, optimal_alpha, Knm, Kmm, y, batch_dict, num_clients, max_iter=10000):
    """
    Run the federated training pipeline.
    """
    x = np.random.randn(num_FEATURES) * 0.01
    error_history = [np.linalg.norm(x - optimal_alpha)]

    for _ in range(max_iter // E):
        local_models = local_training_step(x, Knm, Kmm, y, batch_dict, num_clients, E)
        x = sum(local_models) / len(local_models)
        error_history.append(np.linalg.norm(x - optimal_alpha))
    
    return error_history


def plot_results():
    """
    Load data and plot results for different epoch values.
    """
    #with open("first_database.pkl", "rb") as f:
    with open("second_database.pkl", "rb") as f:
        x,y = pickle.load(f)
        x = np.array(x)
        y = np.array(y)
        n = 100
        m = 10
        a = 5
        print("x shape : ", x.shape)
        print("y shape : ", y.shape)
        x_m_points=np.linspace(-1,1,m)
        sel = [i for i in range(num_SAMPLES)]
        np.random.seed(0)
        data_points = np.random.choice(sel, num_SAMPLES, replace=False)
        #data_points = [i for i in range(n)]
        #ind = [i for i in range(n, m + n)]
        indices = np.random.choice(num_SAMPLES, num_FEATURES, replace=False)
        selected_x = [x[i] for i in indices]
        #selected_x = [x_m_points[i] for i in indices]
        y_n = [y[i] for i in range(num_SAMPLES)]

        # init B
        B_algo = dict()
        for i in range(C):
            B_algo[i] = data_points[i*B:(i+1)*B]

        Knm = compute_kernel_fedavg([i for i in range(num_SAMPLES)], selected_x, x)
        Kmm = compute_kernel_fedavg(indices, selected_x, x)
        #Kmm = compute_kernel_matrix(x_m_points, x_m_points)
        #Knm = []
        #for i in range(a):
        #    Knm.append(compute_kernel_matrix(x[i], x_m_points))
        print("Kmm shape : ", Kmm.shape)   
        print("Knm shape : ", Knm[0].shape)
        alpha_star = np.linalg.inv(nu*np.identity(num_FEATURES)+sigma*sigma*Kmm+Knm.T.dot(Knm)).dot(Knm.T.dot(np.array(y_n)))
        print("alpha_star = ", alpha_star)

        T = 10000
        colors = {1: "royalblue", 5: "orange", 50: "forestgreen"}
        for E in [1, 5, 50]:
            ecart_alpha_star = train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=10000)
            
            x = [k * E for k in range(T//E + 1)]
            
            plt.plot(x, np.array(ecart_alpha_star), label="E = "+str(E), color = colors[E])
            plt.yscale('log')
            plt.xscale('log')

        plt.grid()
        plt.title("Setting B = 20, C = 5, constant learning rate = 0.002")
        plt.legend()
        plt.show()

def plot_results_different_C():
    """
    Load data and plot results for different epoch values.
    """
    with open("first_database.pkl", "rb") as f:
    #with open("second_database.pkl", "rb") as f:
        x,y = pickle.load(f)
        x = np.array(x)
        y = np.array(y)
        n = 100
        m = 10
        a = 5
        E = 5
        print("x shape : ", x.shape)
        print("y shape : ", y.shape)
        for C in [5, 10, 15, 20]:
            x_m_points=np.linspace(-1,1,m)
            sel = [i for i in range(num_SAMPLES)]
            np.random.seed(0)
            data_points = np.random.choice(sel, num_SAMPLES, replace=False)
            #data_points = [i for i in range(n)]
            #ind = [i for i in range(n, m + n)]
            indices = np.random.choice(num_SAMPLES, num_FEATURES, replace=False)
            selected_x = [x[i] for i in indices]
            #selected_x = [x_m_points[i] for i in indices]
            y_n = [y[i] for i in range(num_SAMPLES)]

            # init B
            B_algo = dict()
            for i in range(C):
                B_algo[i] = data_points[i*B:(i+1)*B]

            Knm = compute_kernel_fedavg([i for i in range(num_SAMPLES)], selected_x, x)
            Kmm = compute_kernel_fedavg(indices, selected_x, x)
            #Kmm = compute_kernel_matrix(x_m_points, x_m_points)
            #Knm = []
            #for i in range(a):
            #    Knm.append(compute_kernel_matrix(x[i], x_m_points))
            print("Kmm shape : ", Kmm.shape)   
            print("Knm shape : ", Knm[0].shape)
            alpha_star = np.linalg.inv(nu*np.identity(num_FEATURES)+sigma*sigma*Kmm+Knm.T.dot(Knm)).dot(Knm.T.dot(np.array(y_n)))
            print("alpha_star = ", alpha_star)

            T = 10000
            colors = {5: "royalblue", 10: "orange", 15: "pink", 20: "forestgreen"}
            
            ecart_alpha_star = train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=10000)
            
            x = [k * E for k in range(T//E + 1)]
            
            plt.plot(x, np.array(ecart_alpha_star), label="C = "+str(C), color = colors[C])
            plt.yscale('log')
            plt.xscale('log')

        plt.grid()
        plt.title("Setting E = 5, B = 20, constant learning rate = 0.002")
        plt.legend()
        plt.show()

def plot_results_different_sel_clients():
    """
    Load data and plot results for different epoch values.
    """
    #with open("first_database.pkl", "rb") as f:
    with open("second_database.pkl", "rb") as f:
        x,y = pickle.load(f)
        x = np.array(x)
        y = np.array(y)
        n = 100
        m = 10
        a = 5
        E = 5
        print("x shape : ", x.shape)
        print("y shape : ", y.shape)
        for nb in [0, 1, 2, 3]:
            x_m_points=np.linspace(-1,1,m)
            sel = [i for i in range(num_SAMPLES)]
            np.random.seed(nb)
            data_points = np.random.choice(sel, num_SAMPLES, replace=False)
            #data_points = [i for i in range(n)]
            #ind = [i for i in range(n, m + n)]
            indices = np.random.choice(num_SAMPLES, num_FEATURES, replace=False)
            selected_x = [x[i] for i in indices]
            #selected_x = [x_m_points[i] for i in indices]
            y_n = [y[i] for i in range(num_SAMPLES)]

            # init B
            B_algo = dict()
            for i in range(C):
                B_algo[i] = data_points[i*B:(i+1)*B]

            Knm = compute_kernel_fedavg([i for i in range(num_SAMPLES)], selected_x, x)
            Kmm = compute_kernel_fedavg(indices, selected_x, x)
            #Kmm = compute_kernel_matrix(x_m_points, x_m_points)
            #Knm = []
            #for i in range(a):
            #    Knm.append(compute_kernel_matrix(x[i], x_m_points))
            print("Kmm shape : ", Kmm.shape)   
            print("Knm shape : ", Knm[0].shape)
            alpha_star = np.linalg.inv(nu*np.identity(num_FEATURES)+sigma*sigma*Kmm+Knm.T.dot(Knm)).dot(Knm.T.dot(np.array(y_n)))
            print("alpha_star = ", alpha_star)

            T = 10000
            colors = {0: "royalblue", 1: "orange", 2: "pink", 3: "forestgreen"}
            
            ecart_alpha_star = train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=10000)
            
            x = [k * E for k in range(T//E + 1)]
            
            plt.plot(x, np.array(ecart_alpha_star), label="random selection = "+str(nb), color = colors[nb])
            plt.yscale('log')
            plt.xscale('log')

        plt.grid()
        plt.title("Setting E = 5, B = 20, constant learning rate = 0.002")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    #plot_results()
    #plot_results_different_C()
    plot_results_different_sel_clients()
