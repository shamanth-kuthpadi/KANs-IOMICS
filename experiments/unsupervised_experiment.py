import torch
import os
from kan import *
from utilities.utils import *

base_log_dir = '/Users/shamanthk/Documents/KANs-IOMICS/logs/sweep/unsupervised_experiment'

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_config = {
    # --- KAN __init__ parameters ---
    'width': [6, 1, 1],
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

    # --- model.fit() parameters ---
    'opt': 'LBFGS',
    'steps': 50,
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
    'lr':                    [0.01, 0.1, 1.0],
    'batch':                 [-1, 32, 50],
    'lamb_l1':               [0.0, 0.1, 1.0],
    'lamb_coefdiff':         [0.0, 0.01, 0.1],
    'update_grid':           [True, False],
    'grid_update_num':       [5, 10],
    'start_grid_update_step':[ -1, 0],
    'stop_grid_update_step': [10, 30],
    'symbolic_enabled':      [True, False],
    'affine_trainable':      [True, False],
    'grid_eps':              [0.0, 0.02, 1.0],
    'sp_trainable':          [True, False],
    'sb_trainable':          [True, False],
    'sparse_init':           [False, True],
    'singularity_avoiding':  [False, True],
    'y_th':                  [100.0, 1000.0, 10000.0],
}

dataset = create_dataset(device)

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
        
        print(f"\n=== Sweep {param_name} = {val} ===")

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

        model.fix_symbolic(1,0,0,'gaussian',fit_params_bool=False)

        model(dataset['train_input'])

        history = model.fit(
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
            save_fig=True,
            save_fig_freq=1,
            logger='csv',
            log_output=log_name
        )
