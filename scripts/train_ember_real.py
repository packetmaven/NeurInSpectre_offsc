#!/usr/bin/env python3
"""Train EMBER MLP with real corrected labels."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

def train():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    epochs = 20
    batch_size = 256
    lr = 0.001
    
    print(f"[Train] Device: {device}")
    data_path = Path('./data/ember/ember_2018')
    
    # Load data
    print("[Train] Loading EMBER data...")
    x_size = (data_path / 'X_train.dat').stat().st_size
    n_samples = x_size // (2381 * 4)
    
    X_train = np.memmap(str(data_path / 'X_train.dat'), dtype=np.float32, mode='r', shape=(n_samples, 2381))
    y_train = np.memmap(str(data_path / 'y_train.dat'), dtype=np.float32, mode='r', shape=(n_samples,))
    
    # Filter unlabeled (-1)
    labeled_mask = y_train >= 0
    X_labeled = np.array(X_train[labeled_mask]).copy()
    y_labeled = np.array(y_train[labeled_mask].astype(int)).copy()
    
    print(f"[Train] Filtered: {n_samples:,} -> {len(y_labeled):,} labeled samples")
    print(f"[Train] Class distribution: benign={np.sum(y_labeled==0):,}, malicious={np.sum(y_labeled==1):,}")
    
    # Split train/val 80/20
    n_train = int(0.8 * len(y_labeled))
    indices = np.random.permutation(len(y_labeled))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train_split = X_labeled[train_idx]
    y_train_split = y_labeled[train_idx]
    X_val = X_labeled[val_idx]
    y_val = y_labeled[val_idx]
    
    print(f"[Train] Train: {len(y_train_split):,}, Val: {len(y_val):,}")
    
    # Convert to tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train_split), torch.LongTensor(y_train_split))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = nn.Sequential(
        nn.Linear(2381, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 2)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 500 == 0:
                print(f'  [{epoch+1}/{epochs}] Batch {batch_idx+1}/{len(train_loader)} '
                      f'Loss: {train_loss/(batch_idx+1):.4f} Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f'[Epoch {epoch+1}/{epochs}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_dir = Path('./models/checkpoints')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / 'ember_mlp.pt')
            print(f'  -> Saved checkpoint ({val_acc:.2f}%)')
    
    print(f"\n[Done] Best val accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    train()
