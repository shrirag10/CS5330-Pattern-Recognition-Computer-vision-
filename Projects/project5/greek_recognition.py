# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 3

import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from mnist_recognition import MyNetwork

# Task 3A: Custom Transform for Greek letters (operating on Tensors)
class GreekTransform:
    def __init__(self):
        """
        Constructor for GreekTransform.
        """
        pass

    def __call__(self, img):
        """
        Processes a PyTorch Tensor:
        1. Converts RGB to grayscale (reduces channels from 3 to 1)
        2. Inverts intensities (black letters on white background becomes white on black)
        3. Resizes to 28x28
        """
        # Convert RGB tensor (3, H, W) to grayscale tensor (1, H, W)
        if img.shape[0] == 3:
            img = torchvision.transforms.functional.rgb_to_grayscale(img)
            
        # Invert intensities (ToTensor scales values to [0.0, 1.0])
        img = 1.0 - img
        
        # Resize to 28x28 using bilinear interpolation
        img = torchvision.transforms.functional.resize(img, (28, 28))
        
        return img

# Useful function to generate mock custom Greek letters if none exist
def generate_mock_greek(directory):
    """
    Generates mock custom Greek letters (alpha, beta, gamma)
    to test the evaluation pipeline before the user uploads their own.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Generating mock custom Greek letters in '{directory}'...")
    
    # We will generate black-on-white images (simulating handwriting on a white page)
    # 0: alpha, 1: beta, 2: gamma
    names = ["alpha", "beta", "gamma"]
    for idx, name in enumerate(names):
        img = Image.new("L", (128, 128), color=255)
        draw = ImageDraw.Draw(img)
        
        if idx == 0:  # alpha (fish-like shape)
            draw.arc([30, 40, 90, 88], 30, 330, fill=0, width=8)
            draw.line([35, 45, 98, 85], fill=0, width=8)
            draw.line([35, 83, 98, 43], fill=0, width=8)
        elif idx == 1:  # beta (capital B shape with a tail)
            draw.line([44, 25, 44, 108], fill=0, width=8)
            draw.arc([44, 25, 88, 65], 270, 90, fill=0, width=8)
            draw.arc([44, 63, 88, 103], 270, 90, fill=0, width=8)
        elif idx == 2:  # gamma (y-like loop shape)
            draw.line([34, 25, 64, 70], fill=0, width=8)
            draw.line([94, 25, 64, 70], fill=0, width=8)
            draw.ellipse([50, 65, 78, 103], outline=0, width=8)
            
        img.save(os.path.join(directory, f"custom_{name}.png"))
    print("Mock Greek letters generated successfully.")

# Useful function to train the Greek classification layer
def train_greek(model, device, train_loader, optimizer):
    """
    Trains the model for one epoch.
    Returns:
        Average loss for the epoch.
        Training accuracy (%) for the epoch.
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
        
        # Accumulate statistics
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += len(data)
        
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    return avg_loss, accuracy

# Useful function to preprocess custom Greek letter images
def preprocess_greek_image(filepath):
    """
    Loads custom Greek letter image, converts to grayscale,
    resizes to 28x28, auto-inverts (to white-on-black), and normalizes.
    """
    img = Image.open(filepath).convert('L')
    img_np = np.array(img)
    
    # Auto-inversion check using border pixels
    h, w = img_np.shape
    border_pixels = []
    border_pixels.extend(img_np[0, :])       # Top border
    border_pixels.extend(img_np[h-1, :])     # Bottom border
    for r in range(1, h-1):
        border_pixels.append(img_np[r, 0])   # Left
        border_pixels.append(img_np[r, w-1]) # Right
        
    border_mean = np.mean(border_pixels)
    if border_mean > 127.0:
        img_np = 255 - img_np
        img = Image.fromarray(img_np)
        
    # Resize
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_resized_np = np.array(img_resized)
    
    # Convert to Tensor and normalize
    tensor = torch.tensor(img_resized_np, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0)  # (1, 28, 28)
    tensor = torchvision.transforms.functional.normalize(tensor, (0.1307,), (0.3081,))
    
    return tensor, img_resized_np

# Main execution function
def main(argv):
    """
    Main function to execute the transfer learning and custom evaluation for Task 3.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Task 3A: Instantiate DataLoader for the Greek dataset
    print("Loading Greek letters dataset...")
    greek_dataset_path = "data/greek_train"
    
    if not os.path.exists(greek_dataset_path):
        print(f"Error: Greek dataset directory not found at {greek_dataset_path}.")
        sys.exit(1)
        
    greek_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            greek_dataset_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )
    
    # Task 3B: Load pre-trained MNIST CNN and modify it
    model = MyNetwork().to(device)
    mnist_model_path = "models/mnist_model.pth"
    try:
        model.load_state_dict(torch.load(mnist_model_path, map_location=device))
        print(f"Successfully loaded pre-trained MNIST weights from {mnist_model_path}")
    except FileNotFoundError:
        print(f"Error: Pre-trained MNIST model not found at {mnist_model_path}. Please run mnist_recognition.py first.")
        sys.exit(1)
        
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final classification layer (fc2) with a new Linear layer (50 -> 3)
    model.fc2 = nn.Linear(50, 3).to(device)
    print("Replaced fc2 layer with a new linear layer (50 -> 3 output nodes). Existing weights frozen.")
    
    # Task 3C: Set up optimizer and train only the fc2 layer
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    
    epoch = 0
    losses = []
    accuracies = []
    
    print("\nStarting training on Greek letters...")
    print("-" * 50)
    
    # Loop until we reach 100% training accuracy
    while True:
        epoch += 1
        loss, accuracy = train_greek(model, device, greek_loader, optimizer)
        losses.append(loss)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch:2d}: Loss = {loss:.6f}, Training Accuracy = {accuracy:.2f}%")
        
        if accuracy == 100.0:
            print("-" * 50)
            print(f"Reached 100% training accuracy at epoch {epoch}!")
            break
            
        # Prevent infinite loops in case of training issues
        if epoch >= 100:
            print("-" * 50)
            print("Warning: Reached maximum epoch limit of 100 without hitting 100% accuracy.")
            break
            
    # Task 3D: Save the trained Greek model
    os.makedirs("models", exist_ok=True)
    greek_model_save_path = "models/greek_model.pth"
    torch.save(model.state_dict(), greek_model_save_path)
    print(f"Greek letters transfer model saved successfully to {greek_model_save_path}")
    
    # Save the loss curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Greek Letters Fine-Tuning Loss Curve')
    plt.grid(True)
    plt.savefig("greek_loss_plot.png")
    plt.close()
    print("Saved greek_loss_plot.png")
    
    # Task 3E: Test on Custom Greek Letters
    custom_greek_dir = "data/custom_greek"
    if not os.path.exists(custom_greek_dir) or len([f for f in os.listdir(custom_greek_dir) if f.endswith('.png')]) == 0:
        generate_mock_greek(custom_greek_dir)
        
    custom_files = sorted([f for f in os.listdir(custom_greek_dir) if f.endswith('.png')])
    
    # Load the fine-tuned model for evaluation
    model.eval()
    
    greek_labels = ["alpha", "beta", "gamma"]
    
    print("\nEvaluating custom Greek letters:")
    print("-" * 50)
    
    fig = plt.figure(figsize=(10, 4))
    num_images = min(len(custom_files), 3)
    
    for idx in range(num_images):
        filename = custom_files[idx]
        filepath = os.path.join(custom_greek_dir, filename)
        
        # Preprocess
        tensor, preprocessed_img = preprocess_greek_image(filepath)
        
        # Predict
        with torch.no_grad():
            output = model(tensor.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).item()
            probabilities = torch.exp(output)[0].tolist() # Convert log-probs to probs
            
        print(f"File: {filename}")
        print(f"  Predicted Label: {greek_labels[prediction]} (Index: {prediction})")
        print(f"  Confidence: {probabilities[prediction]:.4f}")
        print("-" * 50)
        
        # Plotting
        plt.subplot(1, 3, idx + 1)
        plt.tight_layout()
        plt.imshow(preprocessed_img, cmap='gray', interpolation='none')
        plt.title(f"Pred: {greek_labels[prediction]}\nFile: {filename}")
        plt.xticks([])
        plt.yticks([])
        
    plt.savefig("custom_greek_predictions.png")
    plt.close()
    print("Saved custom_greek_predictions.png")

if __name__ == "__main__":
    main(sys.argv)
