import pandas as pd
import numpy as np
import torch
from kan import *
import copy
import tempfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

torch.set_default_dtype(torch.float64)
dtype = torch.get_default_dtype()

# Download data: https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb#scrollTo=l10N2ZbHu6Ob
df = pd.read_csv("../knot_data.csv")
df.keys()

X = df[df.keys()[1:]].to_numpy()
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean[np.newaxis,:])/std[np.newaxis,:]

# normalize X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean[np.newaxis,:])/X_std[np.newaxis,:]
input_normalier = [X_mean, X_std]

dataset = {}
num = X.shape[0]
n_feature = X.shape[1]
train_ratio = 0.8
train_id_ = np.random.choice(num, int(num*train_ratio), replace=False)
test_id_ = np.array(list(set(range(num))-set(train_id_)))
dataset['train_input'] = torch.from_numpy(X[train_id_]).type(dtype)
dataset['test_input'] = torch.from_numpy(X[test_id_]).type(dtype)

def construct_contrastive_dataset(tensor):
    y = copy.deepcopy(tensor)
    for i in range(y.shape[1]):
        y[:,i] = y[:,i][torch.randperm(y.shape[0])]
    return y

dataset['contrastive_train_input'] = construct_contrastive_dataset(dataset['train_input'])
dataset['contrastive_test_input'] = construct_contrastive_dataset(dataset['test_input'])

dataset['train_label'] = torch.cat([torch.ones(dataset['train_input'].shape[0],1), torch.zeros(dataset['contrastive_train_input'].shape[0],1)], dim=0).to(device)
dataset['train_input'] = torch.cat([dataset['train_input'], dataset['contrastive_train_input']], dim=0).to(device)

dataset['test_label'] = torch.cat([torch.ones(dataset['test_input'].shape[0],1), torch.zeros(dataset['contrastive_test_input'].shape[0],1)], dim=0).to(device)
dataset['test_input'] = torch.cat([dataset['test_input'], dataset['contrastive_test_input']], dim=0).to(device)

def train_acc():
    return torch.mean(((model(dataset['train_input']) > 0.5) == dataset['train_label']).float())

def test_acc():
    return torch.mean(((model(dataset['test_input']) > 0.5) == dataset['test_label']).float())

num_seeds = 100
lambs = [10e-2, 10e-3]

for seed in range(num_seeds):
    for lamb in range(lambs):
        model = KAN(width=[n_feature,1,1], grid=5, k=3, seed=seed, device=device)
        model.fix_symbolic(1,0,0,'gaussian',fit_params_bool=False)
        model.fit(dataset, lamb=0.001, batch=1024, metrics=[train_acc, test_acc], display_metrics=['train_loss', 'reg', 'train_acc', 'test_acc']);

        # seed = 2024
        model.plot(scale=1.0)

        n = 18
        for i in range(n):
            plt.gcf().get_axes()[0].text(1/(2*n)+i/n-0.005,-0.02,df.keys()[1:][i], rotation=270, rotation_mode="anchor")
            
        print(dataset['train_input'].shape)