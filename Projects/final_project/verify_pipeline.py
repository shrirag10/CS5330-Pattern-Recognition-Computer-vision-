import os
import shutil
import torch
from src.dataset import get_dataset_splits
from src.model import get_model
from src.train import train_model

def run_verification():
    print("="*60)
    print("RUNNING PIPELINE VERIFICATION")
    print("="*60)
    
    data_dir = "/home/shrirag10/Projects/CS5330/Projects/final_project/data"
    temp_results_dir = "/home/shrirag10/Projects/CS5330/Projects/final_project/results/verify_temp"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Test Dataset split and loader
    print("\nTesting Dataset Loader...")
    try:
        train_loader, val_loader, test_loader, classes = get_dataset_splits(
            data_dir=data_dir,
            batch_size=8,  # Small batch size for testing
            num_workers=2,
            seed=42
        )
        print("Dataset loaders created successfully!")
        print(f"Classes found: {classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
        
    # 2. Test Model Factory for all three conditions
    conditions = ['pretrained-frozen', 'random-frozen', 'random-full']
    print("\nTesting Model Factory and parameter configurations...")
    for cond in conditions:
        try:
            model = get_model(condition=cond, num_classes=len(classes), seed=42)
            print(f"Successfully created model for condition: {cond}")
            
            # Check parameter sizes and gradient tracking
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            
            if cond in ['pretrained-frozen', 'random-frozen']:
                # The final block (layer4) has 8,393,728 params
                # The head (fc) has 512 * 6 = 3,072 weights + 6 biases = 3,078 params
                # So trainable params should be 8,396,806.
                expected_trainable = 8396806
                print(f"  Condition: {cond} | Trainable parameters: {trainable_count} (Expected: {expected_trainable}) | Frozen: {frozen_count}")
                assert trainable_count == expected_trainable, f"Trainable params {trainable_count} does not match expected {expected_trainable}"
            else: # random-full
                print(f"  Condition: {cond} | Trainable parameters: {trainable_count} (All) | Frozen: {frozen_count}")
                assert frozen_count == 0, f"Frozen params {frozen_count} is not zero for random-full"
                
        except Exception as e:
            print(f"Error testing model condition {cond}: {e}")
            return False
            
    # 3. Run a quick mock training of 1 epoch for pretrained-frozen
    print("\nTesting Mock Training and early stopping checkpointing...")
    try:
        model = get_model(condition='pretrained-frozen', num_classes=len(classes), seed=42)
        
        # We will use small loaders with only a few batches to speed up the test
        # Let's create subset loaders
        from torch.utils.data import Subset, DataLoader
        
        train_subset = Subset(train_loader.dataset, range(16))
        val_subset = Subset(val_loader.dataset, range(16))
        
        test_train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        test_val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
        history, epochs_trained = train_model(
            model=model,
            train_loader=test_train_loader,
            val_loader=test_val_loader,
            condition='pretrained-frozen',
            results_dir=temp_results_dir,
            seed=42,
            max_epochs=2,
            patience=1,
            lr=1e-3,
            device=device
        )
        
        print("Mock training completed successfully!")
        print(f"Epochs trained: {epochs_trained}")
        print(f"History: {history}")
        
        # Verify checkpoint is saved
        checkpoint_file = os.path.join(temp_results_dir, "best_model_pretrained-frozen_seed42.pth")
        if os.path.exists(checkpoint_file):
            print(f"Checkpoint file saved successfully at {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print(f"Checkpoint contents: Epoch={checkpoint['epoch']}, Val Loss={checkpoint['val_loss']:.4f}, Val Acc={checkpoint['val_acc']*100:.2f}%")
        else:
            print("Checkpoint file not found!")
            return False
            
    except Exception as e:
        print(f"Error in mock training: {e}")
        return False
        
    # Clean up temp folder
    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
        print("\nCleaned up temporary verification files.")
        
    print("\n" + "="*60)
    print("PIPELINE VERIFICATION SUCCESSFUL!")
    print("="*60)
    return True

if __name__ == '__main__':
    run_verification()
