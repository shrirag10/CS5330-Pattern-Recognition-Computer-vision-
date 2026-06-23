# Project 3: Real-Time 2-D Object Recognition

We built a real-time 2-D object recognition system. The program processes frames from a live webcam or static images. It extracts translation, scale, and rotation invariant features. It can build a database of objects and recognize them.

We wrote the core image processing steps from scratch without using OpenCV's built-in thresholding, morphology, or connected components functions.

## Build Instructions

```bash
cd Projects/project3
cmake -S . -B build
cmake --build build -- -j$(nproc)
```

All executables are placed in `build/`.

---

## Task 1, 2, 3 — Thresholding, Morphological Cleanup, Segmentation

Run the recognizer on a single image to see the full pipeline in the three display windows (Original Feed, Thresholded View, Regions Map):

```bash
./build/recognition test_data01/Carrot/IMG_4442.jpg
```

Or run on the bundled dev images in headless batch mode (saves outputs to `saved_thresholds/`):

```bash
./build/recognition --test
```

**Keyboard controls in the display windows:**

| Key | Action |
|-----|--------|
| `q` / Esc | Quit |
| `t` | Toggle threshold overlay in main window |
| `d` | Toggle dynamic (ISODATA) vs manual threshold |
| `i` | Toggle inversion (dark-on-light / light-on-dark) |
| `[` / `]` | Decrease / increase manual threshold by 5 |
| `c` | Toggle morphological cleanup |
| `-` / `=` | Decrease / increase structuring element size |
| `n` / `p` | Next / previous dev image (static mode only) |
| `s` | Save all pipeline stage images to `saved_thresholds/` |

---

## Task 4 — Feature Extraction and Oriented Bounding Box

Covered by the same `recognition` executable above. The Original Feed window overlays the OBB, primary axis arrow, and centroid crosshair on each detected region.

To save pipeline stage images for any image:

```bash
./build/visualize_pipeline --input test_data01/Carrot/IMG_4442.jpg --output report_imgs --prefix carrot
```

This saves six PNGs: original, threshold, cleanup, regions, features, and a side-by-side strip.

---

## Task 5 — Collecting Training Data

Organize training images into subfolders named after each class:

```
train_data01/
  Carrot/
  Clip/
  Glasses/
  PhoneCase/
  Watch/
```

Run the batch trainer to build a CSV feature database:

```bash
./build/train --dataset train_data01 --output database.csv --classical
```

Each row in `database.csv` has the format: `label,elongation,h1,h2`.

To train a CNN embedding database instead:

```bash
./build/train --dataset train_data01 --output database_cnn.csv --cnn --model resnet18.onnx
```

---

## Task 6 — Classifying New Images

The `recognition` executable loads `database.csv` automatically on startup and overlays the predicted label in green on the largest detected region. Run it on any test image:

```bash
./build/recognition test_data01/Glasses/IMG_4477.jpg
```

To run batch classification and print per-image results:

```bash
./build/evaluate --dataset test_data01 --db database.csv --classical
```

---

## Task 7 — Confusion Matrix and Evaluation

Run `evaluate` on a held-out test set to print the confusion matrix and accuracy:

```bash
./build/evaluate --dataset test_data01 --db database.csv --classical
```

To evaluate using CNN embeddings:

```bash
./build/evaluate --dataset test_data01 --db database_cnn.csv --cnn --model resnet18.onnx
```

Output includes per-image `[OK]` / `[MISS]` lines, the full confusion matrix, and overall accuracy.

---

## Task 8 — Demo Video

See `task8.mp4` in this folder. It shows the recognizer running on four test images (Carrot, Glasses, PhoneCase, Watch) with the predicted label overlaid in real time.

---

## Task 9 — ResNet18 DNN Embedding Classifier

Export the ResNet18 ONNX model (requires PyTorch):

```bash
pip install torch torchvision onnx onnxscript
python export_model.py
```

This writes `resnet18.onnx` to the current directory. Then train and evaluate as shown in Tasks 5 and 7 using the `--cnn` flag.

The CNN classifier uses cosine similarity against stored 512-d embeddings. The `recognition` interactive mode will automatically use CNN classification if `resnet18.onnx` is present alongside `database_cnn.csv`.
