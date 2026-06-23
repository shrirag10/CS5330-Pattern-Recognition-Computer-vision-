# Project 3 Report Notes

---

## Task 5 — Collect Training Data

**Overview**
The system collects feature vectors from labeled object images and stores them in a CSV database for use in classification. Training is done in batch mode from a structured folder of images rather than live capture.

**Input Structure**
```
train_data01/
  PhoneCase/   ← folder name is the label
  Glasses/
  Carrot/
  Watch/
  Clip/
```
Each subfolder contains ~8 JPG images of the same object in different poses and orientations.

**Command**
```bash
./build/train --dataset train_data01 --output database.csv
```

**Output**
A CSV file `database.csv` with one row per image:
```
label,elongation,h1,h2
PhoneCase,3.79811,0.211703,0.015242
...
```
Where `elongation`, `h1`, and `h2` are rotation, scale, and translation invariant shape features derived from region moments.

**Analysis**
- PhoneCase produced highly consistent features (elongation ~3.8) across all poses — good candidate for classification
- Glasses showed high variance in elongation (1.3–19.2) due to the arms being open or closed in different images — a known limitation of shape-only features
- Watch also showed high variance due to reflections on the watch face affecting segmentation
- Carrot and Clip produced moderate variance, acceptable for classification

---

## Task 6 — Classify New Images

**Overview**
The system loads the training database and classifies each new image using nearest-neighbor matching with a scaled Euclidean distance metric. For each feature dimension, distances are normalized by the standard deviation across all training entries to prevent any single feature from dominating.

**Distance metric**
```
distance = sum((f1_i - f2_i)^2 / stdev_i^2)
```
The training entry with the smallest distance is returned as the predicted label.

**Command**
```bash
# classify a single image interactively
./build/recognition train_data01/PhoneCase/IMG_4428.jpg

# batch evaluate on test set
./build/evaluate --dataset test_data01 --db database.csv
```

**Result**
The predicted label is overlaid in green text above the object centroid in the Original Feed window. In batch mode, each prediction is printed with OK or MISS alongside the filename.

**Analysis**
- 100% accuracy on the 10-image test set (2 per class)
- Test set is small — results are promising but not conclusive
- Watch/Carrot confusion observed in earlier runs before the dataset was cleaned up — both are elongated objects with similar moment features, illustrating the limitation of shape-only classification

---

## Task 7 — Evaluate System Performance

**Overview**
The system is evaluated on a held-out test set of 2 images per class that were not used during training. Results are summarized in a 5x5 confusion matrix where rows are true labels and columns are predicted labels. The diagonal represents correct classifications.

**Command**
```bash
./build/evaluate --dataset test_data01 --db database.csv
```

---

### Dataset 1 — Unsuitable objects (Chips, Book, Coffee, Tape, Clip)

Objects photographed on white background but with mixed colors, colorful packaging, and non-uniform surfaces.

**Confusion Matrix**
```
                Book    Chips    Clip    Coffee    Tape
------------------------------------------------------
Book              2       0       0        0        0
Chips             2       0       0        0        0
Clip              0       0       2        0        0
Coffee            0       0       0        0        2
Tape              0       0       1        0        1

Accuracy: 5/10 (50%)
```

**Observations**
- Chips confused with Book — both segment as large rectangular blobs
- Coffee confused with Tape — both segment as compact rounded shapes
- The pipeline's shape features cannot distinguish objects that have similar 2D silhouettes
- Color and texture are discarded entirely by the grayscale thresholding step

---

### Dataset 2 — Better suited objects (PhoneCase, Glasses, Carrot, Watch, Clip)

Dark or high-contrast objects photographed on white paper. More suitable for the shape-based pipeline.

**Confusion Matrix**
```
                Carrot    Clip    Glasses    PhoneCase    Watch
--------------------------------------------------------------
Carrot             2       0         0           0          0
Clip               0       2         0           0          0
Glasses            0       0         2           0          0
PhoneCase          0       0         0           2          0
Watch              0       0         0           0          2

Accuracy: 10/10 (100%)
```

**Observations**
- All 5 classes correctly classified
- PhoneCase had the most consistent features (elongation ~3.8) — ideal object for this pipeline
- Glasses had high feature variance across poses due to arms being open/closed — got lucky on test images
- Watch/Carrot confusion observed in earlier cross-dataset runs — both elongated objects with similar moment features
- Key finding: dataset quality and object suitability have a larger impact on accuracy than classifier choice

---

## Task 8 — Demo Video

*(to be filled)*
