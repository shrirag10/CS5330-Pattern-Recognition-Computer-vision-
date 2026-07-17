# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Task 1F

import sys
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Useful function to generate mock handwritten digits if the directory is empty
def generate_mock_digits(directory):
    """
    Generates 10 mock digit images (0-9) using PIL and saves them as PNG files.
    This ensures the pipeline runs out-of-the-box even before the user uploads their photos.
    """
    os.makedirs(directory, exist_ok=True)
    print(f"Generating mock digit images in '{directory}'...")
    
    # We will generate black-on-white images (which mimics photos of white paper with black ink)
    # to test the auto-inversion logic.
    for digit in range(10):
        # Create a 128x128 white image
        img = Image.new("L", (128, 128), color=255)
        draw = ImageDraw.Draw(img)
        
        # Draw digit using lines to look handwritten-like
        if digit == 0:
            draw.ellipse([30, 30, 98, 98], outline=0, width=8)
        elif digit == 1:
            draw.line([64, 25, 64, 103], fill=0, width=8)
        elif digit == 2:
            draw.arc([34, 30, 94, 70], 180, 0, fill=0, width=8)
            draw.line([94, 50, 34, 103], fill=0, width=8)
            draw.line([34, 103, 94, 103], fill=0, width=8)
        elif digit == 3:
            draw.arc([34, 25, 94, 65], 270, 90, fill=0, width=8)
            draw.arc([34, 63, 94, 103], 270, 90, fill=0, width=8)
        elif digit == 4:
            draw.line([34, 25, 34, 69], fill=0, width=8)
            draw.line([34, 69, 94, 69], fill=0, width=8)
            draw.line([74, 25, 74, 103], fill=0, width=8)
        elif digit == 5:
            draw.line([84, 25, 44, 25], fill=0, width=8)
            draw.line([44, 25, 44, 60], fill=0, width=8)
            draw.arc([34, 55, 94, 103], 270, 90, fill=0, width=8)
        elif digit == 6:
            draw.ellipse([34, 60, 94, 103], outline=0, width=8)
            draw.line([44, 25, 34, 60], fill=0, width=8)
        elif digit == 7:
            draw.line([34, 25, 94, 25], fill=0, width=8)
            draw.line([94, 25, 44, 103], fill=0, width=8)
        elif digit == 8:
            draw.ellipse([40, 25, 88, 63], outline=0, width=8)
            draw.ellipse([34, 60, 94, 103], outline=0, width=8)
        elif digit == 9:
            draw.ellipse([34, 25, 94, 68], outline=0, width=8)
            draw.line([88, 55, 88, 103], fill=0, width=8)
            
        img.save(os.path.join(directory, f"digit_{digit}.png"))
    print("Mock digits generated successfully.")

# Useful function to preprocess custom digit images using PIL
def preprocess_image(filepath):
    """
    Reads an image using PIL, converts to grayscale, resizes to 28x28,
    checks if it needs intensity inversion, and normalizes it.
    Returns:
        torch.Tensor of shape (1, 28, 28) ready for model input.
        original preprocessed image (numpy array) for plotting.
    """
    # Open image and convert to grayscale ('L')
    img = Image.open(filepath).convert('L')
    img_np = np.array(img)
    
    # Auto-inversion check:
    # MNIST digits are white (high values) on a black background (low values).
    # Photos of handwritten digits are typically black text on a white page.
    # We analyze the average value of border pixels. If they are mostly bright (> 127),
    # we assume black-on-white and invert the image.
    h, w = img_np.shape
    border_pixels = []
    border_pixels.extend(img_np[0, :])       # Top border
    border_pixels.extend(img_np[h-1, :])     # Bottom border
    for r in range(1, h-1):
        border_pixels.append(img_np[r, 0])   # Left border
        border_pixels.append(img_np[r, w-1]) # Right border
    
    border_mean = np.mean(border_pixels)
    if border_mean > 127.0:
        # Invert: white background becomes black, black writing becomes white
        img_np = 255 - img_np
        img = Image.fromarray(img_np)
        
    # Resize to 28x28 using Lanczos interpolation
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_resized_np = np.array(img_resized)
    
    # Scale to [0.0, 1.0] and convert to PyTorch Tensor
    tensor = torch.tensor(img_resized_np, dtype=torch.float32) / 255.0
    
    # Add channel dimension: (1, 28, 28)
    tensor = tensor.unsqueeze(0)
    
    # Normalize with MNIST training set mean and standard deviation
    tensor = torchvision.transforms.functional.normalize(tensor, (0.1307,), (0.3081,))
    
    return tensor, img_resized_np

# Main function
def main(argv):
    """
    Main function to run predictions on custom handwritten digit images.
    """
    # Check if we can import MyNetwork from mnist_recognition
    # We do it locally to ensure sys.path is correct
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mnist_recognition import MyNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    custom_dir = "data/custom_digits"
    
    # Check if directory contains images. If not, generate mock digits.
    # Note: We check if there are PNG files.
    os.makedirs(custom_dir, exist_ok=True)
    png_files = [f for f in os.listdir(custom_dir) if f.endswith('.png')]
    if len(png_files) == 0:
        generate_mock_digits(custom_dir)
        png_files = sorted([f for f in os.listdir(custom_dir) if f.endswith('.png')])
    else:
        png_files = sorted(png_files)
        
    # Load model and weights
    model = MyNetwork().to(device)
    model_path = "models/mnist_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded trained weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Trained model weights not found at {model_path}.")
        sys.exit(1)
        
    # Load custom images and run predictions
    print("\nEvaluating custom handwritten digits:")
    print("-" * 50)
    
    fig = plt.figure(figsize=(12, 6))
    num_images = min(len(png_files), 10)
    
    for idx in range(num_images):
        filename = png_files[idx]
        filepath = os.path.join(custom_dir, filename)
        
        # Preprocess
        tensor, preprocessed_img = preprocess_image(filepath)
        
        # Predict
        with torch.no_grad():
            output = model(tensor.unsqueeze(0).to(device))
            prediction = output.argmax(dim=1).item()
            probabilities = torch.exp(output)[0].tolist() # Convert log-probs to probs
            
        print(f"File: {filename}")
        print(f"  Predicted Label: {prediction}")
        print(f"  Confidence: {probabilities[prediction]:.4f}")
        print("-" * 50)
        
        # Plotting
        plt.subplot(2, 5, idx + 1)
        plt.tight_layout()
        plt.imshow(preprocessed_img, cmap='gray', interpolation='none')
        plt.title(f"Pred: {prediction}\nFile: {filename}")
        plt.xticks([])
        plt.yticks([])
        
    plt.savefig("custom_digits_predictions.png")
    plt.close()
    print("Saved custom_digits_predictions.png")

if __name__ == "__main__":
    main(sys.argv)
