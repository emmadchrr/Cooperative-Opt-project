import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *

# Global Constants
B = 20
C = 5
E = [1, 5, 50]
learning_rate = 0.002
num_SAMPLES = 100
num_FEATURES = 10
REGULARIZATION = 5
sigma = 0.5
sigma2 = 0.25
nu = 1.0

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

def aggregate_models(local_models):
    """
    Aggregate local models into a single global model.
    """
    return sum(local_models) / len(local_models)

def compute_kernel_fedavg(indices, selected_x, x):
    # Exemple de fonction de calcul de noyau
    selected_x = np.array(selected_x).reshape(-1, x.shape[1])  # Conversion en tableau numpy et ajustement des dimensions
    return np.dot(x[indices], selected_x.T)

def train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter):
    # Exemple de fonction d'entraînement fédéré
    ecart_alpha_star = []
    for _ in range(max_iter):
        ecart_alpha_star.append(np.linalg.norm(alpha_star))
    return ecart_alpha_star

def plot_results():
    num_SAMPLES = 5
    num_FEATURES = 20
    C = 2
    B = 2
    sigma = 0.5
    nu = 1.0
    m = 10  # Nombre de points m

    with open("second_database.pkl", "rb") as f:
        x, y = pickle.load(f)
        
    # Conversion en tableaux numpy
    x = np.array(x)
    y = np.array(y)
    print("x shape : ", x.shape)
    print("y shape : ", y.shape)

    # Génération des points m
    x_m_points = np.linspace(-1, 1, m).reshape(-1, 1)  # Ajustement des dimensions pour correspondre à x
    selected_x = np.hstack([x_m_points] * num_FEATURES)  # Répéter les points pour correspondre au nombre de caractéristiques
    y_n = [y[i] for i in range(num_SAMPLES)]

    # init B
    B_algo = dict()
    for i in range(C):
        B_algo[i] = [j for j in range(i * B, (i + 1) * B)]

    Knm = compute_kernel_fedavg([i for i in range(num_SAMPLES)], selected_x, x)
    Kmm = compute_kernel_fedavg(range(m), selected_x, selected_x)

    print("Kmm shape : ", Kmm.shape)
    print("Knm shape : ", Knm.shape)

    alpha_star = np.linalg.inv(nu * np.identity(m) + sigma * sigma * Kmm + Knm.T.dot(Knm)).dot(Knm.T.dot(np.array(y_n)))
    print("alpha_star = ", alpha_star)

    T = 10000
    colors = {1: "royalblue", 5: "orange", 50: "forestgreen"}
    for E in [1, 5, 50]:
        ecart_alpha_star = train_federated(E, alpha_star, Knm, Kmm, y, B_algo, C, max_iter=T)
        x_vals = [k * E for k in range(len(ecart_alpha_star))]
        plt.plot(x_vals, np.array(ecart_alpha_star), label="E = " + str(E), color=colors[E])
        plt.yscale('log')

    plt.xlabel('Iterations')
    plt.ylabel('Ecart alpha star')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_results()