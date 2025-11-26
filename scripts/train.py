#!/usr/bin/env python3
"""
Train Neural Surrogate for Flash Calciner

Usage:
    python scripts/train.py
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from calciner import (
    CalcinerSimulator,
    SpatiallyAwareDynamics,
    SurrogateModel,
    generate_training_data,
    train_surrogate,
)


def main():
    print("=" * 70)
    print("Neural Surrogate Training for Flash Calciner")
    print("=" * 70)
    
    # Configuration
    N_z = 20
    dt = 0.1
    n_trajectories = 30
    trajectory_length = 30
    n_epochs = 50
    
    # Create simulator
    simulator = CalcinerSimulator(N_z=N_z, dt=dt)
    print(f"\n✓ Simulator: {simulator.state_dim}D state, {simulator.control_dim}D control")
    
    # Generate training data
    print(f"\nGenerating {n_trajectories * trajectory_length} transitions...")
    train_data = generate_training_data(
        simulator,
        n_trajectories=n_trajectories,
        trajectory_length=trajectory_length,
    )
    
    # Split train/val
    n_total = len(train_data)
    n_val = n_total // 5
    indices = np.random.permutation(n_total)
    
    train_dataset = torch.utils.data.Subset(train_data, indices[n_val:])
    val_dataset = torch.utils.data.Subset(train_data, indices[:n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create and train model
    model = SpatiallyAwareDynamics(N_z=N_z)
    print(f"\nTraining {model.__class__.__name__} ({sum(p.numel() for p in model.parameters()):,} params)...")
    
    history = train_surrogate(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        lr=1e-3,
    )
    
    # Save
    norm_params = train_data.get_normalization_params()
    safe_norm_params = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in norm_params.items()}
    
    save_path = Path(__file__).parent.parent / "models" / "surrogate_model.pt"
    save_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': safe_norm_params,
        'model_class': model.__class__.__name__,
        'N_z': N_z,
        'dt': dt,
    }, save_path)
    
    print(f"\n✓ Saved model to {save_path}")
    print(f"✓ Final val loss: {history['val_loss'][-1]:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()

