"""
WHAT: Hyperparameter sweep for training a KAN (Kolmogorov-Arnold Network) classifier on synthetic data.
WHY: To evaluate the effect of various KAN and training hyperparameters on classification performance.
ASSUMES: 
    - Existence of KAN implementation (from `kan` package) and utility functions (`create_dataset_clas`) in `utilities.utils`.
    - Dataset is in a format compatible with KAN training (dict with 'train_input', 'train_label', 'test_input', 'test_label').
    - PyTorch, pandas, seaborn, and matplotlib are installed.
    - Output directory exists or is creatable.
    - If you **do** want the shock coefficient mechanism introduced during training, set `shock_coef=True` in `model.fit()`.
FUTURE IMPROVEMENTS: 
    - Parallelize the hyperparameter sweep.
    - Add logging of model evaluation metrics post-training.
    - Add checkpointing or early stopping.
    - Allow command-line config overrides.
VARIABLES: 
    - `default_config`: Dictionary of baseline hyperparameters for the KAN model and training.
    - `sweep_config`: Dictionary of parameters to sweep and their respective value lists.
    - `dataset`: Dictionary containing train/test splits of inputs and labels.
    - `log_dir`, `log_name`: Paths for saving logs of each hyperparameter configuration.
    - `shock_coef`: Used to enable or disable "shock" regularization during training.
WHO: [S.K.S] 2025/05/31

SAMPLE OUTPUT:
Skipping k = 1 (already exists)
=== Running sweep: lamb = 0.001 ===
...
Logs saved to /logs/sweep/supervised_clas_experiment/lamb/lamb_0.001.csv
"""

# Import statements
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from kan import *  # Contains KAN model implementation
from utilities.utils import *  # Contains `create_dataset_clas`

# Set logging directory and device
base_log_dir = '/Users/shamanthk/Documents/KANs-IOMICS/logs/sweep/supervised_clas_experiment'
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default KAN model and training configuration
default_config = {
    # KAN model hyperparameters
    'width': [2, 2],
    'scale_sp': 1.0,
    'grid': 3,
    'k': 3,
    'mult_arity': 2,
    'noise_scale': 0.3,
    'scale_base_mu': 0.0,
    'scale_base_sigma': 1.0,
    'base_fun': 'silu',
    'symbolic_enabled': True,
    'affine_trainable': False,
    'grid_eps': 0.02,
    'grid_range': [-1, 1],
    'sp_trainable': True,
    'sb_trainable': True,
    'seed': 147,
    'save_act': True,
    'sparse_init': False,

    # Training configuration
    'opt': 'LBFGS',
    'steps': 20,
    'lr': 1.0,
    'batch': -1,
    'lamb': 0.0,
    'lamb_l1': 1.0,
    'lamb_entropy': 2.0,
    'lamb_coef': 0.0,
    'lamb_coefdiff': 0.0,
    'update_grid': True,
    'grid_update_num': 10,
    'start_grid_update_step': -1,
    'stop_grid_update_step': 50,
    'singularity_avoiding': False,
    'y_th': 1000.0,
}

# Hyperparameter sweep configuration
sweep_config = {
    'grid':                  [1, 2, 3, 5, 7],
    'k':                     [1, 2, 3, 5, 7],
    'lamb':                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'lamb_coef':             [1e-1, 5e-1, 1, 1.5, 4.0],
    'lamb_entropy':          [1, 2, 5, 10, 20],
    'scale_base_mu':         [0, 0.5, 1, 2, 3],
    'scale_base_sigma':      [0, 0.5, 1, 2, 3],
    'noise_scale':           [0, 0.5, 1, 2, 3],
    'seed':                  [4, 9, 110, 124, 147],
    'steps':                 [10, 20, 50],
    'lr':                    [0.001, 0.01, 0.1, 1.0, 2.0],
    'batch':                 [-1, 32, 50],
    'lamb_l1':               [0.0, 0.1, 1.0],
    'lamb_coefdiff':         [0.0, 0.01, 0.1],
    'update_grid':           [True, False],
    'grid_update_num':       [5, 10],
    'start_grid_update_step':[ -1, 0],
    'stop_grid_update_step': [10, 30],
    'symbolic_enabled':      [True, False],
    'affine_trainable':      [True, False],
    'grid_eps':              [0.0, 0.001, 0.02, 0.1, 0.5, 1.0],
    'sp_trainable':          [True, False],
    'sb_trainable':          [True, False],
    'sparse_init':           [False, True],
    'singularity_avoiding':  [False, True],
    'y_th':                  [100.0, 1000.0, 10000.0],
}

logs = []

# Load dataset
dataset = create_dataset_clas(device)
dtype = torch.get_default_dtype()

# Sweep over specified parameters
for param_name, values in sweep_config.items():
    for val in values:
        log_dir = os.path.join(base_log_dir, param_name)
        os.makedirs(log_dir, exist_ok=True)
        log_name = os.path.join(log_dir, f"{param_name}_{val}.csv")
        
        if os.path.exists(log_name):
            print(f"Skipping {param_name} = {val} (already exists)")
            continue
        
        config = default_config.copy()
        config[param_name] = val

        print(f"\n=== Running sweep: {param_name} = {val} ===")

        # Instantiate KAN model
        model = KAN(
            width=config['width'],
            scale_sp=config['scale_sp'],
            grid=config['grid'],
            k=config['k'],
            mult_arity=config['mult_arity'],
            noise_scale=config['noise_scale'],
            scale_base_mu=config['scale_base_mu'],
            scale_base_sigma=config['scale_base_sigma'],
            base_fun=config['base_fun'],
            symbolic_enabled=config['symbolic_enabled'],
            affine_trainable=config['affine_trainable'],
            grid_eps=config['grid_eps'],
            grid_range=config['grid_range'],
            sp_trainable=config['sp_trainable'],
            sb_trainable=config['sb_trainable'],
            seed=config['seed'],
            save_act=config['save_act'],
            sparse_init=config['sparse_init'],
            device=device
        )

        # Warm-up forward pass
        model(dataset['train_input'])

        # Define metrics
        def train_acc():
            """
            Computes training accuracy.
            :return: Scalar tensor of training accuracy.
            """
            return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))

        def test_acc():
            """
            Computes test accuracy.
            :return: Scalar tensor of test accuracy.
            """
            return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))
        
        # Train model
        model.fit(
            dataset,
            opt=config['opt'],
            steps=config['steps'],
            lr=config['lr'],
            batch=config['batch'],
            lamb=config['lamb'],
            lamb_l1=config['lamb_l1'],
            lamb_entropy=config['lamb_entropy'],
            lamb_coef=config['lamb_coef'],
            lamb_coefdiff=config['lamb_coefdiff'],
            update_grid=config['update_grid'],
            grid_update_num=config['grid_update_num'],
            start_grid_update_step=config['start_grid_update_step'],
            stop_grid_update_step=config['stop_grid_update_step'],
            singularity_avoiding=config['singularity_avoiding'],
            y_th=config['y_th'],
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics=(train_acc, test_acc),
            save_fig=True,
            save_fig_freq=1,
            logger='csv',
            log_output=log_name,
            shock_coef=False
        )