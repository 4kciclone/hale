#!/bin/bash
set -e
echo "=== HALE Project Setup ==="
echo "Installing PyTorch with ROCm 6.0 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
echo "Installing other dependencies..."
pip install -r requirements.txt
echo "Verifying GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'Memory: {mem:.1f}GB')
"
echo "Setup complete."
