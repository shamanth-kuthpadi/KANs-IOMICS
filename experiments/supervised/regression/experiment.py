"""
WHAT: Hyperparameter sweep for training a KAN (Kolmogorov-Arnold Network) on a supervised regression task.
WHY: To evaluate the effect of various KAN and training hyperparameters on classification performance.
ASSUMES: 
    - The `KAN` model is defined in `kan`, and the `create_dataset_super` function exists in `utilities.utils`.
    - The dataset returned by `create_dataset_super` includes 'train_input', 'train_label', 'test_input', and 'test_label' tensors.
    - PyTorch (with CUDA optionally), OS module, and all referenced utilities are installed and functional.
    - The output directory structure is writable.
    - This script uses regression evaluation based on rounded predictions, assuming a binary-like output format.
    - If you **do** want the shock coefficient mechanism introduced during training, set `shock_coef=True` in `model.fit()`.
FUTURE IMPROVEMENTS:
    - Parallelize the hyperparameter sweep.
    - Add logging of model evaluation metrics post-training.
    - Add checkpointing or early stopping.
    - Allow command-line config overrides.
VARIABLES:
    - `default_config`: Dictionary specifying default hyperparameters for model architecture and optimization.
    - `sweep_config`: Parameter grid to sweep across with different values for each hyperparameter.
    - `dataset`: Contains the full training and test data split loaded once per run.
    - `log_dir`, `log_name`: Paths for saving logs of each hyperparameter configuration.    
    - `shock_coef`: Used to enable or disable "shock" regularization during training.
WHO: [S.K.S] 2025/05/31

SAMPLE OUTPUT:
Skipping k = 2 (already exists)
=== Running sweep: lr = 0.1 ===
Training complete. Logs saved to: /logs/sweep/supervised_reg_experiment/lr/lr_0.1.csv
"""

import torch
import os
from kan import *
from utilities.utils import *
import argparse

parser = argparse.ArgumentParser(description="KAN Supervised Hyperparameter Sweep")
parser.add_argument('--shock', action='store_true', help="Enable shock regularization during training")
args = parser.parse_args()

if args.shock:
    base_log_dir = '/Users/shamanthk/Documents/KANs-IOMICS/logs/shock/supervised_reg_experiment'
else:
    base_log_dir = '/Users/shamanthk/Documents/KANs-IOMICS/logs/sweep/supervised_reg_experiment'

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'scale_sp': 1.0,
    'mult_arity': 2,

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
    'reg_metric': 'edge_forward_spline_n',
    'beta': 3.0,
}

sweep_config = {
    # chosen arbitrarly -- but with the background knowledge of how grids and polynomial function orders operate in B-spline function approximation
    'grid':                  [1, 2, 3, 5, 7],
    'k':                     [1, 2, 3, 5, 7], 
    # regularization parameters chosen arbitrarly as I do not have any intuition on what value work for what setting
    'lamb':                  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'lamb_coef':             [1e-1, 5e-1, 1, 1.5, 4.0],
    'lamb_entropy':          [1, 2, 5, 10, 20],
    # chosen with the knowledge that scale_base_mu should ideally be close to 0 or 1 -- this is typical in most machine learning contexts
    'scale_base_mu':         [0, 0.5, 1, 2, 3],
    # chosen with the knowledge that scale_base_sigma and noise_scale adds noise and so varying values from low noise to high noise would be good for experimentation
    'scale_base_sigma':      [0, 0.5, 1, 2, 3],
    'noise_scale':           [0, 0.5, 1, 2, 3],
    # completely random seed settings -- I actually used a random number generator for this
    'seed':                  [4, 9, 110, 124, 147],
    # arbitrary selections
    'steps':                 [10, 20, 50],
    'lr':                    [0.001, 0.01, 0.1, 1.0, 2.0],
    'batch':                 [-1, 32, 50],
    'lamb_l1':               [0.0, 0.1, 1.0],
    'lamb_coefdiff':         [0.0, 0.01, 0.1],
    # boolean, so only two values
    'update_grid':           [True, False],
    # arbitrary selections
    'grid_update_num':       [5, 10],
    'start_grid_update_step':[ -1, 0],
    'stop_grid_update_step': [10, 30],
    # boolean, so only two values
    'symbolic_enabled':      [True, False],
    'affine_trainable':      [True, False],
    # arbitrary selections
    'grid_eps':              [0.0, 0.001, 0.02, 0.1, 0.5, 1.0],
    # boolean, so only two values
    'sp_trainable':          [True, False],
    'sb_trainable':          [True, False],
    'sparse_init':           [False, True],
    'singularity_avoiding':  [False, True],
    # arbitrary selections
    'y_th':                  [100.0, 1000.0, 10000.0],
    'base_fun':              ['silu', 'identity', 'zero', 'relu', 'tanh', 'sin'],
    'scale_sp':              [0.0, 0.5, 1.0, 2.0, 3.0],
    'mult_arity':            [1, 2, 3, 4, 5],
    'reg_metric':            ['edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'],
    'beta':                  [1.0, 2.0, 3.0, 4.0, 5.0],
}


# Load data once
dataset = create_dataset_super(device)
dtype = torch.get_default_dtype()

for param_name, values in sweep_config.items():
    for val in values:
        log_dir = os.path.join(base_log_dir, param_name)
        os.makedirs(log_dir, exist_ok=True)
        log_name = os.path.join(log_dir, f"{param_name}_{val}.csv")
        
        if os.path.exists(log_name):
            print(f"Skipping {param_name} = {val} (already exists)")
            continue
        # override one param at a time
        config = default_config.copy()
        config[param_name] = val

        print(f"\n=== Running sweep: {param_name} = {val} ===")

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

        model(dataset['train_input'])


        def train_acc():
            preds = torch.round(model(dataset['train_input'])[:,0])
            return torch.mean((preds == dataset['train_label'][:,0]).type(dtype))

        def test_acc():
            preds = torch.round(model(dataset['test_input'])[:,0])
            return torch.mean((preds == dataset['test_label'][:,0]).type(dtype))

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
            metrics=(train_acc, test_acc),
            save_fig=True,
            save_fig_freq=1,
            logger='csv',
            log_output=log_name,
            shock_coef=args.shock
        )
