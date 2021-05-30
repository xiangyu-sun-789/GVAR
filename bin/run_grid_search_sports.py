# This script evaluates GVAR model across a range of hyperparameter values for the specified simulation experiment.
import argparse
import os
import numpy as np
import time
from datetime import date
from sklearn import preprocessing
import pandas as pd
from experimental_utils_sports import run_grid_search

parser = argparse.ArgumentParser(description='Grid search')

# Model specification
parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
parser.add_argument('--K', type=int, default=2, help='Model order (default: 5)')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers (default: 1)')
parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer '
                                                                      '(default: 50)')

# Training procedure
parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size (default: 256)')
parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs to train (default: 10)')
parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 value for the Adam optimiser (default: 0.9)')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 value for the Adam optimiser (default: 0.999)')

# Meta
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--num-sim', type=int, default=1, help='Number of simulations (default: 1)')
parser.add_argument('--use-cuda', type=bool, default=True, help='Use GPU? (default: true)')

# Parsing args
args = parser.parse_args()

datasets = []  # contains the sports data
structures = None  # the true graph, set to None
signed_structures = None

# print(str(args.num_sim) + " " + str(args.experiment) + " datasets...")


fileName = "/Users/shawnxys/Development/Data/preprocessed_causal_sports_data_by_games/17071/features_shots_rewards.csv"

features_shots_rewards_df = pd.read_csv(fileName)
# rename column name
features_shots_rewards_df = features_shots_rewards_df.rename(columns={'reward': 'goal'})

X = features_shots_rewards_df.to_numpy()

# data standardization
scaler = preprocessing.StandardScaler().fit(X)
normalized_X = scaler.transform(X)

print('feature std after standardization: ', normalized_X.std(axis=0))
assert (normalized_X.std(axis=0).round(
    decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

assert normalized_X.shape == (4021, 12)

datasets.append(normalized_X)

variable_names = [s for s in features_shots_rewards_df.columns]
print(variable_names)

lambdas = np.array([0.2])
gammas = np.array([0.5])

# Perform inference
# GVAR model
if not args.use_cuda:
    print("WARNING: GVAR only supports CUDA!")

print("Device:          GPU...")
print("Model:           GVAR...")

run_grid_search(lambdas=lambdas, gammas=gammas, datasets=datasets, K=args.K,
                num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                num_epochs=args.num_epochs, batch_size=args.batch_size, initial_lr=args.initial_lr,
                beta_1=args.beta_1, beta_2=args.beta_2, seed=args.seed, variable_names=variable_names)
