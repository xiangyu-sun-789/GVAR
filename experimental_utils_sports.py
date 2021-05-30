import os
import time
import numpy as np
from utils import save_adjacency_matrix_in_csv, draw_DAGs_using_LINGAM
from datetime import date
from training import training_procedure_trgc


def run_grid_search(lambdas: np.ndarray, gammas: np.ndarray, datasets: list, K: int,
                    num_hidden_layers: int, hidden_layer_size: int, num_epochs: int, batch_size: int,
                    initial_lr: float, beta_1: float, beta_2: float, seed: int, variable_names=None):
    """
    Evaluates GVAR model across a range of hyperparameters.

    @param lambdas: values for the sparsity-inducing penalty parameter.
    @param gammas: values for the smoothing penalty parameter.
    @param datasets: list of time series datasets.
    @param structures: ground truth GC structures.
    @param K: model order.
    @param num_hidden_layers: number of hidden layers.
    @param hidden_layer_size: number of units in a hidden layer.
    @param num_epochs: number of training epochs.
    @param batch_size: batch size.
    @param initial_lr: learning rate.
    @param seed: random generator seed.
    @param signed_structures: ground truth signs of GC interactions.
    """
    # Logging
    logdir = "logs/" + str(date.today()) + "_" + str(round(time.time())) + "_validation_gvar"
    print("Log directory: " + logdir + "/")
    os.mkdir(path=logdir)
    np.savetxt(fname=logdir + "/lambdas.csv", X=lambdas)
    np.savetxt(fname=logdir + "/gammas.csv", X=gammas)

    n_datasets = len(datasets)

    print("Iterating through " + str(len(lambdas)) + " x " + str(len(gammas)) + " grid of parameters...")
    for i in range(len(lambdas)):
        lmbd_i = lambdas[i]
        for j in range(len(gammas)):
            gamma_j = gammas[j]
            print("λ = " + str(lambdas[i]) + "; γ = " + str(gammas[j]) + "; " +
                  str((i * len(gammas) + j) / (len(gammas) * len(lambdas)) * 100) + "% done")

            for l in range(n_datasets):
                d_l = datasets[l]

                # a_hat_l: binary adjacency matrix
                # a_hat_l_: edge strength matrix
                a_hat_l, a_hat_l_, coeffs_full_l = training_procedure_trgc(data=d_l, order=K,
                                                                           hidden_layer_size=hidden_layer_size,
                                                                           end_epoch=num_epochs, lmbd=lmbd_i,
                                                                           gamma=gamma_j, batch_size=batch_size,
                                                                           seed=(seed + i + j),
                                                                           num_hidden_layers=num_hidden_layers,
                                                                           initial_learning_rate=initial_lr,
                                                                           beta_1=beta_1, beta_2=beta_2,
                                                                           verbose=False)

                file_name_binary = logdir + "/estimated_binary_DAG_{}_{}_{}.csv".format(i, j, l)
                file_name_strength = logdir + "/estimated_strength_DAG_{}_{}_{}.csv".format(i, j, l)

                save_adjacency_matrix_in_csv(file_name_binary, a_hat_l * a_hat_l_, variable_names)
                draw_DAGs_using_LINGAM(file_name_binary, a_hat_l * a_hat_l_, variable_names)

                save_adjacency_matrix_in_csv(file_name_strength, a_hat_l_, variable_names)
                draw_DAGs_using_LINGAM(file_name_strength, a_hat_l_, variable_names)
