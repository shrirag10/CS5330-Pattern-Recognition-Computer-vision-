# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 2

import sys
import os
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mnist_recognition import MyNetwork

# Useful function to print network structure and conv1 weights
def print_network_details(model):
    """
    Prints the layer structure of the model and details of the conv1 weights.
    """
    print("\n" + "=" * 50)
    print("Task 2A: Network Architecture Structure")
    print("=" * 50)
    print(model)
    
    print("\n" + "=" * 50)
    print("Task 2B: conv1 Filter Weights and Shapes")
    print("=" * 50)
    weights = model.conv1.weight.data
    print(f"Weight Tensor Shape: {weights.shape}")
    print(f"Number of Filters: {weights.shape[0]}")
    print(f"Channels per Filter: {weights.shape[1]}")
    print(f"Filter Kernel Size: {weights.shape[2]}x{weights.shape[3]}\n")
    
    for i in range(weights.shape[0]):
        print(f"Filter {i} Weights (5x5 matrix):")
        # Print each row of the 5x5 filter weight
        filter_matrix = weights[i, 0].cpu().numpy()
        for row in filter_matrix:
            print("  [" + " ".join(f"{val:6.4f}" for val in row) + "]")
        print("-" * 50)

# Useful function to plot the 10 filters in a 3x4 grid
def visualize_filters(model):
    """
    Plots the ten 5x5 filters of conv1 as subplots in a 3x4 grid.
    Disables tick marks and labels each subplot with its index.
    """
    weights = model.conv1.weight.data.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("conv1 Filter Weights Visualization", fontsize=14)
    
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # Using a diverging color map 'coolwarm' to clearly see positive and negative weights
        plt.imshow(weights[i, 0], cmap='coolwarm', interpolation='none')
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])
        
    # Leave subplots 11 and 12 empty or remove their axes
    for i in range(10, 12):
        ax = plt.subplot(3, 4, i + 1)
        ax.axis('off')
        
    plt.savefig("conv1_filters.png")
    plt.close()
    print("\nSaved conv1_filters.png")

# Useful function to apply the filters to the first MNIST training image
def apply_filters_to_image(model, device):
    """
    Loads the first training image from MNIST, applies the 10 conv1 filters
    using F.conv2d, plots the results in a 3x4 grid, and saves the output.
    """
    # Load first training image
    print("\nLoading MNIST training dataset to extract the first image...")
    train_dataset = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))
    image, label = train_dataset[0]
    
    # Add batch and channel dimensions for F.conv2d: [1, 1, 28, 28]
    input_tensor = image.unsqueeze(0).to(device)
    
    # Extract weights and transfer model to device
    weights = model.conv1.weight.data.to(device)
    
    # Run convolution (using padding=2 to keep size at 28x28)
    with torch.no_grad():
        filtered_outputs = F.conv2d(input_tensor, weights, padding=2)
        
    # Remove batch dimension: shape becomes [10, 28, 28]
    filtered_outputs = filtered_outputs.squeeze(0).cpu().numpy()
    
    # Original image for reference
    original_img = image[0].numpy()
    
    # Plotting original and the 10 filtered images
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"Original Image (Label: {label}) & 10 conv1 Filter Outputs", fontsize=14)
    
    # Plot original image in slot 1
    plt.subplot(3, 4, 1)
    plt.tight_layout()
    plt.imshow(original_img, cmap='gray', interpolation='none')
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    
    # Plot filtered outputs in slots 2 to 11
    for i in range(10):
        plt.subplot(3, 4, i + 2)
        plt.tight_layout()
        plt.imshow(filtered_outputs[i], cmap='gray', interpolation='none')
        plt.title(f"Filter {i} Output")
        plt.xticks([])
        plt.yticks([])
        
    # Leave slot 12 empty
    ax = plt.subplot(3, 4, 12)
    ax.axis('off')
    
    plt.savefig("conv1_filtered_images.png")
    plt.close()
    print("Saved conv1_filtered_images.png")

# Useful function to print the technical description of the filter behavior
def print_filter_analysis():
    """
    Prints a short technical analysis of the learned conv1 filters.
    """
    analysis_text = """
======================================================================
Task 2E: Filter Analysis Discussion
======================================================================
The learned conv1 filters operate as low-level feature extractors:

1. Directional Edge Detectors:
   - Several filters show distinct patterns of positive (red/warm) and
     negative (blue/cool) weights side-by-side. For example, vertical
     boundaries detect horizontal gradients (left-to-right transitions),
     while horizontal boundaries detect vertical gradients.
   - In the filtered output images, these filters produce bright or dark
     edges along the boundaries of the handwritten strokes.

2. Spot/Blob/Corner Detectors (High-Pass/Center-Surround):
   - Some filters have a strong central region of positive or negative
     weights surrounded by the opposite sign. These act as blob or corner
     detectors, highlighting stroke tips or crossings.

3. Low-Pass/Smoothing Filters:
   - A few filters show relatively uniform, low-contrast weight values.
     These act as smoothing or blurring operators, preserving overall shape
     information while reducing high-frequency noise.

Overall, the combination of these edge and blob detectors allows the subsequent
layers to assemble higher-level geometric primitives (loops, lines, corners)
essential for classifying the handwritten digits 0-9.
======================================================================
"""
    print(analysis_text)

# Main function
def main(argv):
    """
    Main function to run the network examination script.
    """
    # Check for CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate network and load trained weights
    model = MyNetwork().to(device)
    model_path = "models/mnist_model.pth"
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weight file not found at {model_path}. Please run mnist_recognition.py first.")
        sys.exit(1)
        
    # Task 2A & 2B: Print structure and weight matrices
    print_network_details(model)
    
    # Task 2C: Visualize weights
    visualize_filters(model)
    
    # Task 2D: Apply filters to training image
    apply_filters_to_image(model, device)
    
    # Task 2E: Print qualitative analysis
    print_filter_analysis()

if __name__ == "__main__":
    main(sys.argv)
