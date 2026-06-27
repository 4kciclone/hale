#!/usr/bin/env python3
import sys
import time
import traceback
sys.path.insert(0, '.')
from config import get_device
from training.trainer import run_experiment
from training.checkpoint import load_results, save_results, already_done

def main():
    device = get_device()
    results = load_results()
    session_start = time.time()
    BENCHMARKS = ['split_mnist', 'sequential_mnist', 'sequential_cifar10', 'sequential_cifar100']
    for dataset_key in BENCHMARKS:
        key = f'{dataset_key}_main'
        if already_done(results, key):
            print(f'[SKIP] {dataset_key} already completed.')
            continue
        elapsed = (time.time() - session_start) / 3600
        print(f'\n{"=" * 60}')
        print(f'BENCHMARK: {dataset_key.upper()}')
        print(f'Session elapsed: {elapsed:.1f}h')
        print(f'{"=" * 60}')
        try:
            result = run_experiment(dataset_key, device=device)
            results[key] = result
            save_results(results)
        except Exception as e:
            print(f'ERROR in {dataset_key}: {e}')
            traceback.print_exc()
            continue
    print(f'\n{"=" * 60}')
    print('PHASE 1 -- FINAL RESULTS')
    print(f'{"=" * 60}')
    print(f'{"Benchmark":<25} | {"Model":<10} | {"Acc%":>6} | {"BWT%":>7} | {"F%":>6}')
    print('-' * 62)
    for bkey in BENCHMARKS:
        r = results.get(f'{bkey}_main')
        if not r:
            print(f'{bkey:<25} | NOT COMPLETED')
            continue
        first = True
        for model in ['hale', 'backprop', 'esn']:
            m = r[model]
            label = bkey if first else ''
            first = False
            print(f'{label:<25} | {model:<10} | {m["avg_acc"] * 100:>5.1f}% | {m["bwt"] * 100:>+6.1f}% | {m["forgetting"] * 100:>5.1f}%')
        print()
    total = (time.time() - session_start) / 3600
    print(f'Total time: {total:.1f}h')

if __name__ == '__main__':
    main()
