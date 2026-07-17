# Project 5: Recognition using Deep Networks

**Author:** Shriman Raghav Srinivasan  
**Class:** CS 5330: Pattern Recognition & Computer Vision  
**Instructor:** Prof. Bruce Maxwell  

---

## Overview

This project implements digit and character recognition using deep learning networks in PyTorch. It covers:
1. Building, training, and testing a custom Convolutional Neural Network (CNN) on the MNIST dataset.
2. Running inference on custom handwritten digits and printing log-probabilities.
3. Examining the CNN's first-layer convolutional filters and mapping responses.
4. Transfer learning on Greek letters (alpha, beta, gamma) by freezing backbones.
5. Building and training a Vision Transformer (ViT) architecture on MNIST.
6. Automating a 50-configuration hyperparameter sweep to evaluate training dynamics.

---

## Repository Structure

The directory contains the following main scripts:
*   `mnist_recognition.py`: Defines the `MyNetwork` CNN, runs the training and testing loop for 5 epochs, and saves weights to `models/mnist_model.pth`.
*   `test_recognition.py`: Loads the pre-trained weights and runs inference on the first 10 MNIST test images, printing log-probabilities and plotting a 3x3 grid.
*   `test_handwritten.py`: Runs inference on preprocessed custom handwritten digit images.
*   `examine_network.py`: Extracts and visualizes `conv1` weights (`conv1_filters.png`) and applied responses on an MNIST image (`conv1_filtered_images.png`).
*   `greek_recognition.py`: Retrains the pre-trained MNIST CNN's classification head on Greek letters.
*   `mnist_transformer.py`: Implements and trains the Vision Transformer (`NetTransformer`) on MNIST.
*   `mnist_hyperparameter_sweep.py`: Runs a randomized search over 50 unique hyperparameter configurations for 2 epochs each, saving outputs to a CSV.
*   `draw_network.py`: Generates the schematic diagram of the CNN model architecture (`network_diagram.png`).
*   `report.tex` / `report.pdf`: The detailed final LaTeX report and compiled PDF document.

---

## Installation & Environment

Make sure you have PyTorch, torchvision, matplotlib, and PIL installed:
```bash
pip install torch torchvision matplotlib pillow
```

To run all scripts, execute them from the project directory:
```bash
python3 mnist_recognition.py
python3 test_recognition.py
python3 test_handwritten.py
python3 examine_network.py
python3 greek_recognition.py
python3 mnist_transformer.py
python3 mnist_hyperparameter_sweep.py
```

---

## Submission Information

*   **Group Members:** Shriman Raghav Srinivasan (Individual Submission)
*   **Time Travel Days Used:** 0 days
*   **Video Links:** N/A (none submitted)
*   **Acknowledge AI Usage:** Claude AI was used for debugging only.
