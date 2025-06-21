## This file is for the relation extraction heuristic formal implementation --> take a look at the MultKAN.py file and the method get_visible_edges function
from kan import *
import torch
import copy
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ===== Dataset =====
def create_dataset(train_num=500, test_num=500):
    def generate_contrastive(x):
        batch = x.shape[0]
        x[:,2] = torch.exp(torch.sin(torch.pi * x[:,0]) + x[:,1]**2)
        x[:,3] = x[:,4]**3

        def corrupt(tensor):
            y = copy.deepcopy(tensor)
            for i in range(y.shape[1]):
                y[:,i] = y[:,i][torch.randperm(y.shape[0])]
            return y

        x_cor = corrupt(x)
        x = torch.cat([x, x_cor], dim=0)
        y = torch.cat([torch.ones(batch,), torch.zeros(batch,)], dim=0)[:,None]
        return x, y

    x = torch.rand(train_num, 6) * 2 - 1
    x_train, y_train = generate_contrastive(x)

    x = torch.rand(test_num, 6) * 2 - 1
    x_test, y_test = generate_contrastive(x)

    return {
        'train_input': x_train.to(device),
        'test_input': x_test.to(device),
        'train_label': y_train.to(device),
        'test_label': y_test.to(device)
    }

# ===== Experiment =====
num_seeds = 4
results = []
found_relations = []

for seed in range(num_seeds):
    torch.manual_seed(seed)
    print(f"\nRunning seed {seed}...")

    model = KAN(width=[6,1,1], grid=3, k=3, seed=seed, device=device,
                noise_scale=2, scale_base_sigma=3, grid_eps=1)

    dataset = create_dataset()

    model(dataset['train_input'])
    model.fix_symbolic(1,0,0,'gaussian', fit_params_bool=False)
    model(dataset['train_input'])

    model.fit(dataset, opt="LBFGS", steps=50, lamb=0.002, lamb_entropy=10.0,
              lamb_coef=1.0, grid_update_num=5, stop_grid_update_step=30)
    model.plot(in_vars=[r'$x_{}$'.format(i) for i in range(1,7)])

    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)

    significant_features = model.get_visible_edges(require_mask=False)
    
    print(f"Seed {seed} significant features: {significant_features}")