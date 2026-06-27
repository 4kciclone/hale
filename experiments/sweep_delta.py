#!/usr/bin/env python3
import sys
import traceback
sys.path.insert(0, '.')
from config import get_device
from training.trainer import run_experiment
from training.checkpoint import load_results, save_results, already_done

def main():
    device = get_device()
    results = load_results()
    BENCHMARKS = ['split_mnist', 'sequential_mnist', 'sequential_cifar10', 'sequential_cifar100']
    DELTAS = [0.001, 0.01, 0.1, 1.0]
    for dataset_key in BENCHMARKS:
        for delta in DELTAS:
            key = f'{dataset_key}_delta_{delta}'
            if already_done(results, key):
                print(f'[SKIP] {key}')
                continue
            print(f'\nRunning {dataset_key} delta={delta}')
            try:
                r = run_experiment(dataset_key, delta=delta, device=device)
                results[key] = r
                save_results(results)
            except Exception as e:
                print(f'ERROR: {e}')
                traceback.print_exc()
                continue
    print('\n-- DELTA SWEEP RESULTS --')
    for dataset_key in BENCHMARKS:
        print(f'\n{dataset_key}:')
        print(f'  {"delta":>8} | {"HALE Acc%":>10} | {"HALE F%":>8}')
        for delta in DELTAS:
            r = results.get(f'{dataset_key}_delta_{delta}')
            if r:
                print(f'  {delta:>8.3f} | {r["hale"]["avg_acc"] * 100:>9.1f}% | {r["hale"]["forgetting"] * 100:>7.1f}%')

if __name__ == '__main__':
    main()
