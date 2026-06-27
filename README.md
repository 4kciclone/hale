# HALE — Hierarchical Adaptive Local-learning Engine

A biologically-inspired continual learning architecture combining:
- **Equilibrium Propagation (EP)** blocks with local energy-based learning
- **Heterogeneous Reservoir Computing** with multi-timescale dynamics
- **Neuromodulatory Critic** for surprise-gated plasticity
- **Closed-Form Null-space Regularization (CFNR)** for catastrophic forgetting prevention
- **Online RLS Readout** (second-order, no backprop)

## Project Structure

```
hale_project/
├── setup.sh              # ROCm environment setup
├── requirements.txt      # Python dependencies
├── config.py             # All hyperparameters and dataset configs
├── models/
│   ├── ep_block.py       # EPBlock — Layer 1
│   ├── reservoir.py      # HeterogeneousReservoir — Layer 2
│   ├── critic.py         # NeuromdulatoryCtric — Layer 3
│   ├── cfnr.py           # RecursiveNullSpace — Layer 4
│   └── hale.py           # HALE — full model
├── baselines/
│   ├── backprop.py       # StandardBackprop MLP baseline
│   └── esn.py            # ClassicESN baseline
├── data/
│   └── datasets.py       # SequentialDataset universal loader
├── training/
│   ├── metrics.py        # BWT, Forgetting (Lopez-Paz & Ranzato, 2017)
│   ├── trainer.py        # run_experiment() universal runner
│   └── checkpoint.py     # JSON results accumulation
├── experiments/
│   ├── phase1_main.py    # Main Phase 1 (4 benchmarks)
│   ├── sweep_delta.py    # δ sweep experiment
│   └── ablation_depth.py # L=1 vs L=3 vs L=5 ablation
└── run_phase1.sh         # Single command to run everything
```

## Quick Start (RunPod MI300X)

```bash
cd hale_project
bash setup.sh          # Install PyTorch with ROCm 6.0
bash run_phase1.sh     # Run all Phase 1 experiments
```

## Benchmarks

| Dataset              | Tasks | Classes/Task | Epochs |
|----------------------|-------|--------------|--------|
| Split MNIST          | 5     | 2            | 20     |
| Sequential MNIST     | 5     | 2            | 20     |
| Sequential CIFAR-10  | 5     | 2            | 20     |
| Sequential CIFAR-100 | 10    | 10           | 20     |

## Metrics

Following Lopez-Paz & Ranzato (NeurIPS 2017):
- **Avg Acc**: Mean accuracy on all tasks after training on all tasks
- **BWT**: Backward Transfer — negative = forgetting
- **Forgetting**: Mean positive memory loss only

## References

- Scellier & Bengio (2017). Equilibrium Propagation.
- Jaeger (2001). Echo State Networks.
- Lopez-Paz & Ranzato (2017). Gradient Episodic Memory for Continual Learning.
