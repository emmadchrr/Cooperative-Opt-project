import numpy as np
import matplotlib.pyplot as plt
import random
from utils import *


def grad_fedavg(alpha, sigma2, nu, x_batch, y_batch, x_selected, B):
    
    K_mm = compute_kernel_matrix(x_selected,x_selected)
    
    K_bm = compute_kernel_matrix(x_batch,x_selected)
    terme_1 = (1/5)*(1/B)*(sigma2/2)*K_mm@alpha
    
    terme_2 = 0.5 *(1/B)* K_bm.T @ (K_bm @ alpha - y_batch)
    terme_3 = (nu/2)*(1/B)*alpha
    
    return terme_1 + terme_2 + terme_3
    


def FedAvg(X, Y, B, E, C, x_m_points, lr, sigma2,nu, nb_epochs):
    batch_size = len(X[0])//B if len(X[0])//B > 0 else 1
    X = np.array(X)
    alpha = np.zeros((len(x_m_points),))
    n_iter = 0
    val_g = []
    for i in range(nb_epochs):
        
        clients = random.sample([i for i in range(len(X))], C)
        client_updates = []
        for i in clients:
            x = alpha
            data_client = X[i]
            y_client = Y[i]
            for e in range(E):
                b = np.random.randint(B)
                batch_x = data_client[b*batch_size:(b+1)*batch_size] if b<len(data_client)//batch_size else X[b*batch_size:]
                #print(batch_x.shape)
                batch_y = y_client[b*batch_size:(b+1)*batch_size]
                grad = grad_fedavg(x, sigma2, nu, batch_x, batch_y, x_m_points, B)
                x -= lr*grad
            client_updates.append(x)
        alpha = np.sum(np.array(client_updates), axis=0)/C
        val_g.append(f_bis(alpha,x_m_points,np.array(X),np.array(Y),sigma2))
        n_iter += 1
    return alpha, val_g


if __name__ == "__main__":
    # Load data
    with open('second_database.pkl', 'rb') as f:
        x, y = pickle.load(f)
    
    n, m, a = 100, 10, 5 
    sigma2 = 0.25
    nu = 1
    lr = 0.01
    nb_epochs = 1000
    E = 10
    C = 5
    B = 20
    
    x_m_points=np.linspace(-1,1,m)
    K_mm = compute_kernel_matrix(x_m_points,x_m_points)
    K_nm = compute_kernel_matrix(np.array(x).flatten(),x_m_points)
    alpha_star = compute_alpha_star(K_mm, K_nm, np.array(y).flatten(), sigma2, nu)
    g_star = f_bis(alpha_star,x_m_points,np.array(x).flatten(),np.array(y).flatten(),sigma2)
    
    
    # Run FedAvg algorithm
    alpha_optim, val_g = FedAvg(x, y, B, E, C, x_m_points, lr, sigma2, nu, nb_epochs)
    
    # # Plot the results
    plt.plot(abs(val_g -g_star), label='FedAvg')
    plt.xlabel('Epochs')
    plt.ylabel('abs(g _ g*)')
    plt.title(f'Evolution of the absolute error, E = {E}, C = {C}, B = {B}')
    plt.show()
    
    ## varying C
    # C_values = [1,2,3,4,5]
    # g_values = []
    # for C in C_values:
    #     alpha_optim, val_g = FedAvg(x, y, B, E, C, x_m_points, lr, sigma2, nu, nb_epochs)
    #     g_values.append(val_g)
        
    # # plot the 3 curves
    # plt.figure(figsize=(10, 6))
    # for i, C in enumerate(C_values):
    #     plt.plot(abs(g_values[i] - g_star), label=f'C = {C}')
    # plt.xlabel('Epochs')
    # plt.ylabel('abs(g _ g*)')
    # plt.title(f'Evolution of the absolute error for different C values')
    # plt.tight_layout()
    # plt.legend()
    # plt.show()