import numpy as np

def compute_metrics(A, n_tasks):
    K = n_tasks - 1
    final_accs = [A[K][j] for j in range(n_tasks) if A[K][j] > 0]
    avg_acc = float(np.mean(final_accs)) if final_accs else 0.0
    bwt_vals, forget_vals = [], []
    for j in range(K):
        if A[j][j] > 0:
            bwt_vals.append(A[K][j] - A[j][j])
            forget_vals.append(max(A[j][j] - A[K][j], 0.0))
    bwt = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    forgetting = float(np.mean(forget_vals)) if forget_vals else 0.0
    return avg_acc, bwt, forgetting

def print_results_table(dataset_key, results_dict):
    print(f'\n-- {dataset_key} --')
    print(f'{"Model":<14} | {"Avg Acc":>8} | {"BWT":>8} | {"Forgetting":>10}')
    print('-' * 50)
    for name, m in results_dict.items():
        print(f'{name:<14} | {m[0]*100:>7.1f}% | {m[1]*100:>+7.1f}% | {m[2]*100:>9.1f}%')
