# Project 3: Real-Time 2-D Object Recognition

We built a real-time 2-D object recognition system. The program processes frames from a live webcam or static images. It extracts translation, scale, and rotation invariant features. It can build a database of objects and recognize them.

We wrote the core image processing steps from scratch. We did not use OpenCV's built-in thresholding, morphology, or connected components functions.

## Features Implemented

* **Grayscale Conversion and Blurring**: Custom grayscale conversion with saturation discount. Custom 3x3 box blur filter from scratch to reduce noise.
* **Custom Binary Thresholding**: Compares pixel values against a threshold. Supports dynamic threshold computation using the iterative ISODATA method.
* **Custom Morphological Filtering**: Custom erosion, dilation, opening, and closing. We implemented these operations using pixel loops to clean up noise and fill internal holes.
* **Connected Components Segmentation**: Custom 8-connected queue-based flood fill. Prunes small regions under 500 pixels. Generates contiguous label IDs.
* **Feature Extraction**: Custom calculations for centroid, central moments, primary axis orientation, and elongation. Builds a feature vector with elongation and Hu moments h1 and h2.
* **GUI Training Mode**: Pauses camera frames and opens a text input overlay on screen when you press the record key. Saves labels and features directly to a CSV file.

## Build Instructions

Follow these commands to configure and compile the program:

```bash
cd Projects/project3
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Run Instructions

You can run the program in three modes:

1. **Live Camera Mode**: Launches with your default webcam.
   ```bash
   ./recognition
   ```
2. **Static Image Mode**: Loads a specific image from the development set.
   ```bash
   ./recognition ../data/img1p3.png
   ```
3. **Headless Verification Mode**: Headless batch execution. It runs all 5 development images and saves output masks to the `saved_thresholds` directory.
   ```bash
   ./recognition --test
   ```

## Keyboard Controls

Use these keys in the display windows to interact with the program:

* **q** or **Esc**: Quit the program.
* **t**: Toggle showing the thresholded binary mask in the main feed window.
* **d**: Toggle between dynamic ISODATA thresholding and manual thresholding.
* **i**: Toggle inverting the binary threshold (light-on-dark versus dark-on-light).
* **[** / **]** or **,** / **.**: Decrease or increase manual threshold value.
* **c**: Toggle morphological cleanup filtering.
* **-** / **=**: Decrease or increase the morphological structuring element size.
* **n** / **p**: Move to the next or previous image (only in static image directory mode).
* **r**: Enter training mode. Freezes the frame and opens the GUI text prompt. Type your label and press Enter to save to the database, or Esc to cancel.
* **s**: Capture a screenshot. Saves the original frame, raw binary, cleaned mask, region color map, and moment overlays.
