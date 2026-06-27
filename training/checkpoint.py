import json
import os

RESULTS_PATH = './results/hale_phase1_results.json'

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}

def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {RESULTS_PATH}')

def already_done(results, key):
    return key in results
