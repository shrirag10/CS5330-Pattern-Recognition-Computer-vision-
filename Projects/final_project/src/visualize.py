import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(results_dir, seeds=[42, 100, 2026]):
    """
    Plots the validation loss and validation accuracy learning curves for all conditions and seeds.
    Saves the figures as results/loss_curves.png and results/accuracy_curves.png.
    """
    conditions = ['pretrained-frozen', 'random-frozen', 'random-full']
    colors = {
        'pretrained-frozen': '#1f77b4',  # Slate Blue
        'random-frozen': '#ff7f0e',      # Amber Orange
        'random-full': '#2ca02c'         # Emerald Green
    }
    
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    
    for condition in conditions:
        for seed in seeds:
            history_path = os.path.join(results_dir, f"history_{condition}_seed{seed}.json")
            if not os.path.exists(history_path):
                continue
                
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            epochs = range(1, len(history['val_loss']) + 1)
            
            # Loss curve
            ax_loss.plot(epochs, history['val_loss'], label=f"{condition} (seed {seed})", 
                         color=colors[condition], alpha=0.8, linestyle='--' if seed != 42 else '-')
            
            # Accuracy curve
            ax_acc.plot(epochs, [x * 100 for x in history['val_acc']], label=f"{condition} (seed {seed})", 
                        color=colors[condition], alpha=0.8, linestyle='--' if seed != 42 else '-')
            
    # Style loss plot
    ax_loss.set_title("Validation Loss Curves across Conditions and Seeds", fontsize=14, fontweight='bold', pad=15)
    ax_loss.set_xlabel("Epochs", fontsize=12)
    ax_loss.set_ylabel("Validation Loss (Cross-Entropy)", fontsize=12)
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    
    # Clean legend to avoid duplicates but keep colors
    handles, labels = ax_loss.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Filter legend to only show conditions, not every seed
    legend_labels = {cond: by_label[f"{cond} (seed 42)"] for cond in conditions if f"{cond} (seed 42)" in by_label}
    ax_loss.legend(legend_labels.values(), legend_labels.keys(), loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    loss_fig_path = os.path.join(results_dir, "loss_curves.png")
    fig_loss.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig_loss)
    print(f"Saved validation loss curves to {loss_fig_path}")
    
    # Style accuracy plot
    ax_acc.set_title("Validation Accuracy Curves across Conditions and Seeds", fontsize=14, fontweight='bold', pad=15)
    ax_acc.set_xlabel("Epochs", fontsize=12)
    ax_acc.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax_acc.grid(True, linestyle=':', alpha=0.6)
    
    handles, labels = ax_acc.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_labels = {cond: by_label[f"{cond} (seed 42)"] for cond in conditions if f"{cond} (seed 42)" in by_label}
    ax_acc.legend(legend_labels.values(), legend_labels.keys(), loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    
    acc_fig_path = os.path.join(results_dir, "accuracy_curves.png")
    fig_acc.savefig(acc_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig_acc)
    print(f"Saved validation accuracy curves to {acc_fig_path}")

def plot_confusion_matrices(results_dir, classes):
    """
    Plots the confusion matrix heatmap for the representative seed of each condition.
    Saves the figure as results/confusion_matrices.png.
    """
    metrics_path = os.path.join(results_dir, "test_evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Cannot plot confusion matrices.")
        return
        
    with open(metrics_path, 'r') as f:
        results = json.load(f)
        
    conditions = ['pretrained-frozen', 'random-frozen', 'random-full']
    
    # Set up matplotlib subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for idx, condition in enumerate(conditions):
        if condition not in results or 'summary' not in results[condition]:
            continue
            
        summary = results[condition]['summary']
        closest_seed = str(summary['closest_seed'])
        
        # Retrieve the confusion matrix
        cm = np.array(results[condition][closest_seed]['confusion_matrix'])
        # Normalize by row (actual class count) to get percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        
        ax.set_title(f"{condition}\n(Seed {closest_seed})", fontsize=13, fontweight='bold', pad=10)
        
        # Add labels
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        
        ax.set_ylabel('True Label', fontsize=11, labelpad=5)
        ax.set_xlabel('Predicted Label', fontsize=11, labelpad=5)
        
        # Annotate cell values: raw count and percentage
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm_normalized[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)",
                        ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')
                
    # Add a global colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    fig.suptitle("Test Confusion Matrices for Representative Runs", fontsize=16, fontweight='bold', y=1.02)
    
    output_path = os.path.join(results_dir, "confusion_matrices.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix heatmaps to {output_path}")

if __name__ == '__main__':
    RESULTS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
    CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    plot_learning_curves(RESULTS_DIR)
    plot_confusion_matrices(RESULTS_DIR, CLASSES)
