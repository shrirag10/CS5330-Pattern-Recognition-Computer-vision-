# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 1E

import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from mnist_recognition import MyNetwork

# Useful function to run model evaluation on the first 10 test examples
def evaluate_first_10(model, device, test_loader):
    """
    Loads the test set, runs inference on the first 10 images,
    prints the 10 output values (formatted to 2 decimal places),
    the predicted index, and the correct label.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Get the first batch from the test loader
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    print("\nEvaluating the first 10 examples in the test set:")
    print("-" * 80)
    
    with torch.no_grad():
        for i in range(10):
            # Take a single image, add a batch dimension, and move to device
            img = images[i].unsqueeze(0).to(device)
            label = labels[i].item()
            
            # Forward pass
            output = model(img)
            
            # Format raw outputs (log-probabilities) to 2 decimal places
            outputs_formatted = [f"{val:.2f}" for val in output[0].tolist()]
            
            # Prediction is the index of the max output value
            prediction = output.argmax(dim=1).item()
            
            print(f"Example {i+1}:")
            print(f"  Outputs: {outputs_formatted}")
            print(f"  Prediction (Max Output Index): {prediction}")
            print(f"  Correct Label: {label}")
            print("-" * 80)

# Useful function to plot the first 9 test examples in a 3x3 grid
def plot_first_9_predictions(model, device, test_loader):
    """
    Plots the first 9 test digits as a 3x3 grid with the model's prediction above each.
    """
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(10, 10))
    with torch.no_grad():
        for i in range(9):
            img = images[i].unsqueeze(0).to(device)
            output = model(img)
            prediction = output.argmax(dim=1).item()
            
            plt.subplot(3, 3, i + 1)
            plt.tight_layout()
            # Show image (original dimensions 1x28x28)
            plt.imshow(images[i][0], cmap='gray', interpolation='none')
            plt.title(f"Prediction: {prediction}\nLabel: {labels[i].item()}")
            plt.xticks([])
            plt.yticks([])
            
    plt.savefig("predictions_grid.png")
    plt.close()
    print("Saved predictions_grid.png")

# Main function
def main(argv):
    """
    Main function to run the Task 1E evaluation script.
    """
    # Check for CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the MNIST test set (no shuffling, so it matches Task 1 exactly)
    print("Loading MNIST test dataset...")
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=False)
        
    # Instantiate model and load the trained weights
    model = MyNetwork().to(device)
    model_path = "models/mnist_model.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded trained weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model weight file not found at {model_path}. Please run mnist_recognition.py first.")
        sys.exit(1)
        
    # Evaluate first 10 examples
    evaluate_first_10(model, device, test_loader)
    
    # Plot first 9 predictions in a 3x3 grid
    plot_first_9_predictions(model, device, test_loader)

if __name__ == "__main__":
    main(sys.argv)
