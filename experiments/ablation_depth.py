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
    BENCHMARKS = ['sequential_cifar10', 'sequential_cifar100']
    DEPTHS = [1, 3, 5]
    for dataset_key in BENCHMARKS:
        for L in DEPTHS:
            key = f'{dataset_key}_L{L}'
            if already_done(results, key):
                print(f'[SKIP] {key}')
                continue
            print(f'\nRunning {dataset_key} L={L}')
            try:
                r = run_experiment(dataset_key, L=L, device=device)
                results[key] = r
                save_results(results)
            except Exception as e:
                print(f'ERROR: {e}')
                traceback.print_exc()
                continue
    print('\n-- DEPTH ABLATION RESULTS --')
    for dataset_key in BENCHMARKS:
        print(f'\n{dataset_key}:')
        print(f'  {"L":>4} | {"HALE Acc%":>10} | {"HALE F%":>8}')
        for L in DEPTHS:
            r = results.get(f'{dataset_key}_L{L}')
            if r:
                print(f'  {L:>4} | {r["hale"]["avg_acc"] * 100:>9.1f}% | {r["hale"]["forgetting"] * 100:>7.1f}%')

if __name__ == '__main__':
    main()
