# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 5

import os
import sys
import time
import random
import csv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

# Configurable CNN model structure
class ConfigurableNetwork(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, fc_nodes, dropout_rate, activation_type):
        super(ConfigurableNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        
        # Compute flattened features size after two 5x5 Conv layers and two 2x2 MaxPool layers
        # 28x28 -> Conv1 -> 24x24 -> MaxPool -> 12x12
        # 12x12 -> Conv2 -> 8x8 -> MaxPool -> 4x4
        self.flat_features = conv2_channels * 16
        
        self.fc1 = nn.Linear(self.flat_features, fc_nodes)
        self.fc2 = nn.Linear(fc_nodes, 10)
        
        # Select activation
        if activation_type == 'relu':
            self.act = F.relu
        elif activation_type == 'gelu':
            self.act = F.gelu
        elif activation_type == 'leaky_relu':
            self.act = F.leaky_relu
        else:
            self.act = F.relu
            
    def forward(self, x):
        x = self.act(F.max_pool2d(self.conv1(x), 2))
        x = self.act(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.flat_features)
        x = self.act(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# Train function for a configuration
def train_config(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
    return total_loss / total, 100. * correct / total

# Test function for a configuration
def test_config(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    return total_loss / total, 100. * correct / total

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset in memory to make sweeps fast
    print("Loading MNIST datasets...")
    train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))
    test_dataset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                              ]))
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Search Space definition
    space = {
        'lr': [0.1, 0.05, 0.01, 0.005, 0.001],
        'batch_size': [32, 64, 128],
        'optimizer': ['SGD', 'AdamW'],
        'dropout': [0.0, 0.25, 0.5],
        'conv1_channels': [8, 12, 16],
        'conv2_channels': [16, 24, 32],
        'fc_nodes': [32, 64, 128],
        'activation': ['relu', 'gelu', 'leaky_relu']
    }
    
    # Generate 50 unique random configurations
    random.seed(42)
    torch.manual_seed(42)
    
    configs = []
    seen = set()
    while len(configs) < 50:
        cfg = {k: random.choice(v) for k, v in space.items()}
        # Create a unique key to prevent duplicates
        key = tuple(cfg[k] for k in sorted(cfg.keys()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
            
    print(f"Generated {len(configs)} unique hyperparameter configurations.")
    
    # Open CSV for logging results
    csv_file = "hyperparameter_sweep_results.csv"
    csv_headers = ["Config_ID", "LR", "Batch_Size", "Optimizer", "Dropout", "Conv1_Channels", "Conv2_Channels", "FC_Nodes", "Activation", "Train_Loss", "Train_Acc", "Test_Loss", "Test_Acc", "Time_Secs"]
    
    results = []
    
    print("\nStarting automated hyperparameter sweep...")
    print("-" * 100)
    
    for idx, cfg in enumerate(configs, 1):
        # Create loader for this specific batch size
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        
        # Instantiate model
        model = ConfigurableNetwork(
            conv1_channels=cfg['conv1_channels'],
            conv2_channels=cfg['conv2_channels'],
            fc_nodes=cfg['fc_nodes'],
            dropout_rate=cfg['dropout'],
            activation_type=cfg['activation']
        ).to(device)
        
        # Instantiate optimizer
        if cfg['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.5)
            
        # Train for 2 epochs and track time
        start_time = time.time()
        
        train_loss, train_acc = 0, 0
        for epoch in range(1, 3):
            train_loss, train_acc = train_config(model, device, train_loader, optimizer)
            
        test_loss, test_acc = test_config(model, device, test_loader)
        elapsed = time.time() - start_time
        
        # Record result
        run_res = {
            "Config_ID": idx,
            "LR": cfg['lr'],
            "Batch_Size": cfg['batch_size'],
            "Optimizer": cfg['optimizer'],
            "Dropout": cfg['dropout'],
            "Conv1_Channels": cfg['conv1_channels'],
            "Conv2_Channels": cfg['conv2_channels'],
            "FC_Nodes": cfg['fc_nodes'],
            "Activation": cfg['activation'],
            "Train_Loss": train_loss,
            "Train_Acc": train_acc,
            "Test_Loss": test_loss,
            "Test_Acc": test_acc,
            "Time_Secs": elapsed
        }
        results.append(run_res)
        
        print(f"Run {idx:2d}/50 | LR: {cfg['lr']:.4f} | Batch: {cfg['batch_size']:3d} | Opt: {cfg['optimizer']:5s} | Drop: {cfg['dropout']:.2f} | "
              f"Conv: ({cfg['conv1_channels']},{cfg['conv2_channels']}) | Act: {cfg['activation']:10s} | Test Acc: {test_acc:.2f}% | Time: {elapsed:.2f}s")
              
    print("-" * 100)
    
    # Save results to CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved all results to {csv_file}")
    
    # Load with pandas for sorting and analysis
    df = pd.DataFrame(results)
    
    # Print Top 5
    top_5 = df.sort_values(by="Test_Acc", ascending=False).head(5)
    print("\nTop 5 Configurations (Highest Test Accuracy):")
    print(top_5[["Config_ID", "LR", "Batch_Size", "Optimizer", "Dropout", "Conv1_Channels", "Conv2_Channels", "Activation", "Test_Acc"]].to_string(index=False))
    
    # Print Bottom 5
    bottom_5 = df.sort_values(by="Test_Acc", ascending=True).head(5)
    print("\nBottom 5 Configurations (Lowest Test Accuracy):")
    print(bottom_5[["Config_ID", "LR", "Batch_Size", "Optimizer", "Dropout", "Conv1_Channels", "Conv2_Channels", "Activation", "Test_Acc"]].to_string(index=False))
    
    # Plotting Sweep Analysis (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning Rate vs Test Accuracy (color-coded by Optimizer)
    for opt in df['Optimizer'].unique():
        sub = df[df['Optimizer'] == opt]
        axs[0, 0].scatter(sub['LR'], sub['Test_Acc'], label=opt, alpha=0.8, edgecolors='black', s=80)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('Learning Rate (log scale)')
    axs[0, 0].set_ylabel('Test Accuracy (%)')
    axs[0, 0].set_title('Learning Rate vs. Test Accuracy')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--")
    
    # Plot 2: Optimizer vs Test Accuracy (Box plot)
    opt_data = [df[df['Optimizer'] == opt]['Test_Acc'].values for opt in df['Optimizer'].unique()]
    axs[0, 1].boxplot(opt_data, labels=df['Optimizer'].unique())
    axs[0, 1].set_ylabel('Test Accuracy (%)')
    axs[0, 1].set_title('Optimizer Impact on Test Accuracy')
    axs[0, 1].grid(True, ls="--")
    
    # Plot 3: Dropout Rate vs Test Accuracy (Scatter plot)
    axs[1, 0].scatter(df['Dropout'], df['Test_Acc'], color='purple', alpha=0.7, edgecolors='black', s=80)
    axs[1, 0].set_xlabel('Dropout Rate')
    axs[1, 0].set_ylabel('Test Accuracy (%)')
    axs[1, 0].set_title('Dropout Rate vs. Test Accuracy')
    axs[1, 0].grid(True, ls="--")
    
    # Plot 4: Activation Type vs Test Accuracy (Box plot)
    act_types = df['Activation'].unique()
    act_data = [df[df['Activation'] == act]['Test_Acc'].values for act in act_types]
    axs[1, 1].boxplot(act_data, labels=act_types)
    axs[1, 1].set_ylabel('Test Accuracy (%)')
    axs[1, 1].set_title('Activation Function Impact')
    axs[1, 1].grid(True, ls="--")
    
    plt.tight_layout()
    plt.savefig("sweep_analysis.png")
    plt.close()
    print("Saved sweep_analysis.png")

if __name__ == "__main__":
    main()
