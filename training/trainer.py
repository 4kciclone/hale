import sys
import time
import traceback
import torch
sys.path.insert(0, '.')

def to_scalar_label(y, device):
    if not isinstance(y, torch.Tensor):
        return torch.tensor(y, dtype=torch.long, device=device)
    return y.squeeze().long().to(device)

def run_experiment(dataset_key, config=None, delta=0.01, L=None, device=None):
    from config import HALE_CONFIG, DATASET_CONFIGS, get_device
    from models.hale import HALE
    from baselines.backprop import StandardBackprop
    from baselines.esn import ClassicESN
    from data.datasets import build_loaders
    from training.metrics import compute_metrics, print_results_table
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw): return it
    if device is None:
        device = get_device()
    if config is None:
        config = HALE_CONFIG.copy()
    config['delta'] = delta
    if L is not None:
        config['L'] = L
    dcfg = DATASET_CONFIGS[dataset_key]
    n_tasks = dcfg['n_tasks']
    n_cls = dcfg['n_classes_per_task']
    d_in = dcfg['d_in']
    T = dcfg['T']
    n_epochs = dcfg['n_epochs']
    train_loaders, test_loaders = build_loaders(dataset_key, dcfg)
    hale = HALE(config, n_classes=n_cls, d_in=d_in, device=device)
    bp = StandardBackprop(d_in=d_in, T=T, n_classes=n_cls, device=device)
    esn = ClassicESN(d_in=d_in, d_r=512, n_classes=n_cls, device=device)
    A_hale = [[0.0] * n_tasks for _ in range(n_tasks)]
    A_bp = [[0.0] * n_tasks for _ in range(n_tasks)]
    A_esn = [[0.0] * n_tasks for _ in range(n_tasks)]
    t0 = time.time()
    for k in range(n_tasks):
        elapsed = (time.time() - t0) / 60
        print(f'\n[{dataset_key}] Task {k + 1}/{n_tasks} | elapsed: {elapsed:.1f}min')
        hale.reset_reservoir_states()
        esn.reset()
        for ep in range(n_epochs):
            bar = tqdm(train_loaders[k], desc=f'ep{ep + 1}/{n_epochs}', leave=False)
            for x_seq, y in bar:
                x_seq = x_seq.to(device, non_blocking=True)
                try:
                    loss, correct, M_t = hale.full_train_step(x_seq, y)
                    bp.train_step(x_seq, y)
                    esn.train_sequence(x_seq, y)
                    if hasattr(bar, 'set_postfix'):
                        bar.set_postfix(loss=f'{loss:.3f}', M_t=f'{M_t:.2f}')
                except Exception:
                    traceback.print_exc()
                    continue
        hale.task_boundary()
        for j in range(k + 1):
            A_hale[k][j] = hale.evaluate(test_loaders[j])
            A_bp[k][j] = bp.evaluate(test_loaders[j])
            A_esn[k][j] = esn.evaluate(test_loaders[j])
        hale.save_checkpoint(f'./results/{dataset_key}_task{k}.pt', k, {'A_hale': A_hale, 'A_bp': A_bp, 'A_esn': A_esn})
    hale_m = compute_metrics(A_hale, n_tasks)
    bp_m = compute_metrics(A_bp, n_tasks)
    esn_m = compute_metrics(A_esn, n_tasks)
    print_results_table(dataset_key, {'HALE': hale_m, 'Backprop': bp_m, 'ESN': esn_m})
    total_min = (time.time() - t0) / 60
    return {
        'dataset': dataset_key, 'delta': delta, 'L': config['L'],
        'hale': {'avg_acc': hale_m[0], 'bwt': hale_m[1], 'forgetting': hale_m[2]},
        'backprop': {'avg_acc': bp_m[0], 'bwt': bp_m[1], 'forgetting': bp_m[2]},
        'esn': {'avg_acc': esn_m[0], 'bwt': esn_m[1], 'forgetting': esn_m[2]},
        'A_hale': A_hale, 'A_bp': A_bp, 'A_esn': A_esn,
        'elapsed_min': total_min,
    }
