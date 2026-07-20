import os
import json
import time
import torch
from src.dataset import get_dataset_splits
from src.model import get_model
from src.train import train_model

def run_all_experiments(data_dir, results_dir, seeds=[42, 100, 2026], max_epochs=100, patience=5, lr=1e-3, batch_size=32, num_workers=4):
    """
    Runs the full grid of experiments: 3 conditions x 3 seeds.
    Saves checkpoints, training logs, and overall summaries to the results directory.
    
    Args:
        data_dir (str): Directory containing the dataset splits.
        results_dir (str): Directory to save output files.
        seeds (list): List of seeds to run.
        max_epochs (int): Maximum training epochs per run.
        patience (int): Early stopping patience epochs.
        lr (float): Base learning rate for SGD.
        batch_size (int): DataLoader batch size.
        num_workers (int): DataLoader num_workers.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    conditions = ['pretrained-frozen', 'random-frozen', 'random-full']
    
    # Track summary stats for final output
    summary = {}
    
    total_start_time = time.time()
    
    # Load data loaders (the stratified validation split is determined by a fixed seed
    # in the split function to ensure consistency, but loaders can be re-created if necessary.
    # We will instantiate them once here since they don't depend on the run's random seed).
    print("Loading data splits...")
    train_loader, val_loader, test_loader, classes = get_dataset_splits(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=42 # Fixed seed for stratified split to keep validation split identical
    )
    
    for condition in conditions:
        summary[condition] = {}
        for seed in seeds:
            print("="*60)
            print(f"RUNNING EXPERIMENT: Condition={condition} | Seed={seed}")
            print("="*60)
            
            run_start_time = time.time()
            
            # 1. Initialize model
            model = get_model(condition=condition, num_classes=len(classes), seed=seed)
            
            # 2. Train model with early stopping
            history, epochs_trained = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                condition=condition,
                results_dir=results_dir,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                device=device
            )
            
            run_duration = time.time() - run_start_time
            
            # 3. Load checkpoint info to retrieve best epoch info
            checkpoint_path = os.path.join(results_dir, f"best_model_{condition}_seed{seed}.pth")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            best_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['val_loss']
            best_val_acc = checkpoint['val_acc']
            
            # Save history JSON file
            history_path = os.path.join(results_dir, f"history_{condition}_seed{seed}.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
                
            # Log results in summary dict
            summary[condition][str(seed)] = {
                'epochs_trained': epochs_trained,
                'best_epoch': best_epoch,
                'best_val_loss': float(best_val_loss),
                'best_val_acc': float(best_val_acc),
                'duration_seconds': run_duration
            }
            
            print(f"\nExperiment Finished: Condition={condition} | Seed={seed}")
            print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"Best Val Acc: {best_val_acc*100:.2f}% | Run duration: {run_duration:.1f}s")
            
    # Save global summary file
    summary_path = os.path.join(results_dir, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    total_duration = time.time() - total_start_time
    print("="*60)
    print(f"ALL EXPERIMENTS COMPLETE. Total duration: {total_duration/60:.2f} minutes.")
    print("="*60)
    
if __name__ == '__main__':
    # Script can be run standalone as well
    DATA_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/data"
    RESULTS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
    run_all_experiments(DATA_DIR, RESULTS_DIR)
