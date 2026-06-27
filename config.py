import torch
import os

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'Device: {device} | GPU: {name} | Memory: {mem:.1f}GB')
    else:
        device = torch.device('cpu')
        print('Device: CPU')
    return device

HALE_CONFIG_PHASE1 = {
    # Proven config — matches Kaggle results
    'L': 3,
    'J': 3,
    'd_r': 128,
    'd_s': 128,
    'alpha_list': [0.1, 0.5, 0.9],
    'rho_list':   [0.5, 0.85, 0.97],
    'beta':       0.1,
    'gamma':      0.1,
    'eta':        1e-3,
    'N_free':     5,
    'N_nudge':    10,
    'kappa':      2.0,
    'lam':        0.01,
    'eps':        1e-6,
    'delta':      0.001,   # tighter CFNR — best from delta sweep
    'clip_alpha': 5.0,
    'rls_delta':  1.0,
    'rls_lambda': 1.0,
}

HALE_CONFIG_PHASE2 = {
    # Large config — for transformer experiments (Phase 2/3)
    'L': 5,
    'J': 5,
    'd_r': 512,
    'd_s': 256,
    'alpha_list': [0.1, 0.3, 0.5, 0.7, 0.9],
    'rho_list':   [0.5, 0.7, 0.85, 0.9, 0.97],
    'beta':       0.1,
    'gamma':      0.1,
    'eta':        1e-3,
    'N_free':     5,
    'N_nudge':    10,
    'kappa':      2.0,
    'lam':        0.01,
    'eps':        1e-6,
    'delta':      0.001,
    'clip_alpha': 5.0,
    'rls_delta':  1.0,
    'rls_lambda': 1.0,
}

# Default for Phase 1
HALE_CONFIG = HALE_CONFIG_PHASE1

DATASET_CONFIGS = {
    'split_mnist': {'dataset': 'MNIST', 'n_tasks': 5, 'n_classes_per_task': 2, 'train_samples': None, 'test_samples': None, 'n_epochs': 20, 'd_in': 28, 'T': 28, 'description': 'Split MNIST: 5 tasks x 2 classes'},
    'sequential_mnist': {'dataset': 'MNIST', 'n_tasks': 5, 'n_classes_per_task': 2, 'train_samples': None, 'test_samples': None, 'n_epochs': 20, 'd_in': 28, 'T': 28, 'description': 'Sequential MNIST: 5 tasks x 2 classes'},
    'sequential_cifar10': {'dataset': 'CIFAR10', 'n_tasks': 5, 'n_classes_per_task': 2, 'train_samples': None, 'test_samples': None, 'n_epochs': 20, 'd_in': 96, 'T': 32, 'description': 'Sequential CIFAR-10: 5 tasks x 2 classes'},
    'sequential_cifar100': {'dataset': 'CIFAR100', 'n_tasks': 10, 'n_classes_per_task': 10, 'train_samples': None, 'test_samples': None, 'n_epochs': 20, 'd_in': 96, 'T': 32, 'description': 'Sequential CIFAR-100: 10 tasks x 10 classes'},
}

RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)
