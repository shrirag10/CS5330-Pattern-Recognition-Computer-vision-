import os
import json
import numpy as np
import torch
from src.dataset import get_dataset_splits
from src.model import get_model

def compute_classification_metrics(targets, preds, num_classes=6):
    """
    Computes confusion matrix, per-class precision, recall, and F1-score.
    Implements calculations in vanilla numpy to avoid sklearn dependency.
    """
    targets = np.array(targets)
    preds = np.array(preds)
    
    # 1. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
            
    # 2. Per-class metrics
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        # Avoid division by zero
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
        
    return cm.tolist(), precision, recall, f1

def evaluate_all_models(data_dir, results_dir, seeds=[42, 100, 2026], num_classes=6):
    """
    Evaluates all 9 trained models on the held-out test split.
    Calculates overall accuracy, means, standard deviations, and detailed
    per-class metrics for the seed closest to the mean of each condition.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    conditions = ['pretrained-frozen', 'random-frozen', 'random-full']
    
    print("Loading test dataset loader...")
    _, _, test_loader, classes = get_dataset_splits(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        seed=42
    )
    
    test_results = {}
    
    for condition in conditions:
        test_results[condition] = {}
        accuracies = []
        
        for seed in seeds:
            checkpoint_path = os.path.join(results_dir, f"best_model_{condition}_seed{seed}.pth")
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                continue
                
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = get_model(condition=condition, num_classes=num_classes, seed=seed)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = outputs.max(1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Compute accuracy
            correct = sum(p == t for p, t in zip(all_preds, all_targets))
            acc = correct / len(all_targets)
            accuracies.append(acc)
            
            # Compute full metrics for this seed
            cm, prec, rec, f1_val = compute_classification_metrics(all_targets, all_preds, num_classes)
            
            test_results[condition][str(seed)] = {
                'accuracy': float(acc),
                'confusion_matrix': cm,
                'precision': [float(x) for x in prec],
                'recall': [float(x) for x in rec],
                'f1': [float(x) for x in f1_val]
            }
            print(f"Evaluated Condition: {condition} | Seed: {seed} | Test Accuracy: {acc*100:.2f}%")
            
        # Calculate statistics
        if accuracies:
            mean_acc = float(np.mean(accuracies))
            std_acc = float(np.std(accuracies))
            
            # Find the seed closest to the mean
            closest_seed = None
            min_diff = float('inf')
            for seed in seeds:
                if str(seed) in test_results[condition]:
                    diff = abs(test_results[condition][str(seed)]['accuracy'] - mean_acc)
                    if diff < min_diff:
                        min_diff = diff
                        closest_seed = seed
                        
            test_results[condition]['summary'] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'closest_seed': closest_seed
            }
            print(f"Condition {condition} summary: Mean={mean_acc*100:.2f}%, Std={std_acc*100:.2f}%, Closest Seed={closest_seed}")
            
    # Save the test evaluation results
    output_path = os.path.join(results_dir, "test_evaluation_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(test_results, f, indent=4)
        
    print(f"\nSaved all test metrics to {output_path}")
    
    # Generate summary printout table
    print("\n" + "="*80)
    print("FINAL TEST ACCURACY SUMMARY TABLE")
    print("="*80)
    print(f"{'Condition':<25} | {'Mean Acc (%)':<15} | {'Std Dev (%)':<15} | {'Best Seed':<12}")
    print("-"*80)
    for cond in conditions:
        if cond in test_results and 'summary' in test_results[cond]:
            m = test_results[cond]['summary']['mean_accuracy'] * 100
            s = test_results[cond]['summary']['std_accuracy'] * 100
            cs = test_results[cond]['summary']['closest_seed']
            print(f"{cond:<25} | {m:<15.2f} | {s:<15.2f} | {cs:<12}")
    print("="*80)
    
    return test_results

if __name__ == '__main__':
    DATA_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/data"
    RESULTS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
    evaluate_all_models(DATA_DIR, RESULTS_DIR)
