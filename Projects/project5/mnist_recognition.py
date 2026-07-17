# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 1

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Class definition for the CNN model
class MyNetwork(nn.Module):
    def __init__(self):
        """
        Constructor for MyNetwork. Defines the layers:
        - conv1: 10 5x5 filters
        - conv2: 20 5x5 filters
        - conv2_drop: 0.5 dropout rate
        - fc1: linear layer with 50 nodes
        - fc2: linear layer with 10 nodes (output classes)
        """
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass for the network
    def forward(self, x):
        """
        Computes forward pass:
        - Conv1 -> MaxPool2d (2x2) -> ReLU
        - Conv2 -> Dropout -> MaxPool2d (2x2) -> ReLU
        - Flatten
        - FC1 -> ReLU
        - FC2 -> LogSoftmax
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# Useful function to train the network for one epoch
def train_network(model, device, train_loader, optimizer, epoch, train_losses, train_accuracies):
    """
    Trains the model for one epoch.
    Calculates batch-wise training loss and accumulates overall epoch statistics.
    """
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)
        
        # Print progress periodically
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                  
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Train Epoch {epoch} Summary: Average Loss: {avg_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)")
    return avg_loss, accuracy

# Useful function to test/evaluate the network
def test_network(model, device, test_loader, test_losses, test_accuracies):
    """
    Evaluates the model on the test dataset.
    Calculates average test loss and test accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)
            
    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    
    print(f"Test Epoch Summary: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n")
    return test_loss, accuracy

# Useful function to plot the first 6 test digits
def plot_first_six(test_loader):
    """
    Plots the first six digits from the test set as a 2x3 grid and saves the image.
    """
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title(f"Label: {labels[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.savefig("first_6_test_digits.png")
    plt.close()
    print("Saved first_6_test_digits.png")

# Main function of the training script
def main(argv):
    """
    Main function to run the network training and evaluation pipeline.
    """
    # Fix random seed for repeatability
    torch.manual_seed(42)
    
    # Check for CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    n_epochs = 5
    
    # Load the datasets
    print("Loading MNIST dataset...")
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=False)
        
    # Plot the first six test digits
    plot_first_six(test_loader)
    
    # Instantiate network and optimizer
    model = MyNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Lists to store metrics over epochs
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Initial testing step before training
    print("Initial baseline evaluation on test set:")
    test_network(model, device, test_loader, test_losses, test_accuracies)
    
    # Training and testing loop
    for epoch in range(1, n_epochs + 1):
        train_network(model, device, train_loader, optimizer, epoch, train_losses, train_accuracies)
        test_network(model, device, test_loader, test_losses, test_accuracies)
        
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/mnist_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}")
    
    # Plot training and testing loss
    # X-axis will be epoch numbers
    epochs_range = list(range(1, n_epochs + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot([0] + epochs_range, test_losses, color='red', marker='o', label='Test Loss')
    plt.plot(epochs_range, train_losses, color='blue', marker='x', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()
    print("Saved loss_plot.png")
    
    # Plot training and testing accuracy
    plt.figure(figsize=(10, 5))
    plt.plot([0] + epochs_range, test_accuracies, color='red', marker='o', label='Test Accuracy')
    plt.plot(epochs_range, train_accuracies, color='blue', marker='x', label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    plt.close()
    print("Saved accuracy_plot.png")

if __name__ == "__main__":
    main(sys.argv)
