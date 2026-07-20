import os
import time
import torch
import torch.nn as nn
from src.model import enforce_bn_eval_for_frozen_layers

def train_model(model, train_loader, val_loader, condition, results_dir, seed, 
                max_epochs=100, patience=5, lr=1e-3, device='cuda'):
    """
    Trains the ResNet-18 model under the specified condition with early stopping.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Loader for training data split.
        val_loader (DataLoader): Loader for validation data split.
        condition (str): Training condition ('pretrained-frozen', 'random-frozen', 'random-full').
        results_dir (str): Directory to save model checkpoints and logs.
        seed (int): Random seed of the current run.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for val loss improvement before stopping.
        lr (float): Base learning rate for SGD.
        device (str): Device to run on ('cuda' or 'cpu').
        
    Returns:
        history (dict): Dictionary containing epoch-by-epoch training and validation loss/accuracy.
        epochs_trained (int): Total number of epochs run before early stopping or max epochs.
    """
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, f"best_model_{condition}_seed{seed}.pth")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Standard SGD optimizer using recommended parameters: learning rate 1e-3, momentum 0.9, weight decay 1e-4
    # Only pass parameters that require gradient updates (theta_t)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-4)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epochs_trained = 0
    
    print(f"\nStarting training for condition: {condition} | Seed: {seed}")
    print(f"Device: {device} | Max epochs: {max_epochs} | Early stopping patience: {patience}")
    print(f"Trainable parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Frozen parameters count: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
    
    for epoch in range(1, max_epochs + 1):
        epochs_trained = epoch
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        # Enforce Batch Normalization layers in theta_f to stay in evaluation mode
        enforce_bn_eval_for_frozen_layers(model)
        
        running_train_loss = 0.0
        running_train_correct = 0
        total_train_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            running_train_correct += predicted.eq(targets).sum().item()
            total_train_samples += inputs.size(0)
            
        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = running_train_correct / total_train_samples
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                running_val_correct += predicted.eq(targets).sum().item()
                total_val_samples += inputs.size(0)
                
        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = running_val_correct / total_val_samples
        
        # Save metrics to history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:02d}/{max_epochs:02d} - Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")
        
        # --- EARLY STOPPING CHECK ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': epoch_val_acc,
                'seed': seed,
                'condition': condition
            }, checkpoint_path)
            # print("  [Checkpoint Saved]")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No validation loss improvement for {patience} epochs.")
                break
                
    print(f"Finished training. Saved best checkpoint to: {checkpoint_path}")
    return history, epochs_trained
