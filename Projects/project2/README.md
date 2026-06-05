## Project 2 implementation details

---

## Project Structure

```
project2/
├── CMakeLists.txt
├── main.cpp
├── Featurizer.h              # Abstract base class for feature extraction
├── FeaturizerImpl.h          # BaselineFeaturizer implementation
├── SimilarityScoring.h       # Abstract base class for distance scoring
├── SimilarityScoringImpl.h   # SSDScoring, HistogramIntersectionScoring, MultiHistogramScoring implementations
├── olympus/                  # Full image database (1107 images)
└── build/                    # Compiled binaries
```

---

## Dataset Setup

Download the image database and place all images in the `olympus/` folder. This serves as the full search database.

---

## Task 1: Baseline Matching (7x7 center patch + SSD)

### How it works

1. Load the target image and extract its feature vector using `BaselineFeaturizer`
2. For each image in the database directory (skipping the target itself), extract its feature vector
3. Compute the SSD score between the target and each database image
4. Sort all results by score (ascending) and print the top 5

### Files involved

| File | Role |
|------|------|
| `Featurizer.h` | Abstract base class with `featurize()` and `featurizeAll()` |
| `FeaturizerImpl.h` | `BaselineFeaturizer` — extracts 7x7 center patch as a 147-value BGR vector |
| `SimilarityScoring.h` | Abstract base class with `score()` |
| `SimilarityScoringImpl.h` | `SSDScoring` — computes Sum of Squared Differences between two feature vectors |
| `main.cpp` | Loads target image, scans database dir, scores all images, sorts and prints top 5 |

### Feature extraction (`BaselineFeaturizer`)

Extracts the 7x7 pixel patch centered in the image. Each pixel has 3 channels (BGR), giving a **147-value feature vector**.

```
center = (width/2, height/2)
patch  = pixels from (cx-3, cy-3) to (cx+3, cy+3)
```

### Distance metric (`SSDScoring`)

Sum of Squared Differences between two feature vectors `a` and `b`:

```
SSD = (a[0]-b[0])² + (a[1]-b[1])² + ... + (a[146]-b[146])²
```

A score of 0 means the patches are identical.

### Build

```bash
cd Projects/project2
mkdir build && cd build
cmake ..
make
```

### Run

```bash
./match <target_image> <database_dir> baseline
```

### Example

```bash
./match ../olympus/pic.1016.jpg ../olympus baseline
```

### Example output

```
Top 5 matches for: pic.1016.jpg (method: baseline)
1. pic.0986.jpg  (score: 14049)
2. pic.0641.jpg  (score: 21756)
3. pic.0547.jpg  (score: 49703)
4. pic.1013.jpg  (score: 51539)
5. pic.0233.jpg  (score: 55806)
```

---

## Task 2: Histogram Matching (3D color histogram + histogram intersection)

### How it works

1. Load the target image and compute its 3D color histogram using `HistogramFeaturizer`
2. For each image in the database (skipping the target), compute its 3D histogram
3. Compute histogram intersection between the target and each database image
4. Sort by score (ascending, since intersection is negated) and print the top 5

### Feature extraction (`HistogramFeaturizer`)

Builds a **3D histogram** over all 3 color channels (B, G, R) jointly with 16 bins per channel:

```
bins per channel = 16
total cells      = 16 × 16 × 16 = 4096
bin width        = 256 / 16 = 16 intensity levels per bin
```

Each cell `hist[bBin][gBin][rBin]` holds the fraction of pixels with that exact color combination. The histogram is normalized by total pixel count so images of different sizes are comparable.

### Distance metric (`HistogramIntersectionScoring`)

For two normalized histograms `a` and `b`:

```
intersection = sum of min(a[i], b[i]) for all i
score        = -intersection  (negated so lower = more similar)
```

Score range is [-1, 0]. A score of -1 means the histograms are identical.

### Run

```bash
./match <target_image> <database_dir> histogram
```

### Example

```bash
./match ../olympus/pic.0164.jpg ../olympus histogram
```

### Example output

```
Top 5 matches for: pic.0164.jpg (method: histogram)
1. pic.0110.jpg  (score: -0.388535)
2. pic.0092.jpg  (score: -0.360647)
3. pic.1032.jpg  (score: -0.337152)
4. pic.0426.jpg  (score: -0.30144)
5. pic.0080.jpg  (score: -0.283518)
```

---

## Task 3: Multi-histogram Matching (3x3 spatial grid + weighted histogram intersection)

### How it works

1. Divide both images into a **3×3 grid** of 9 regions
2. For each region, compute a 3D color histogram (8 bins per channel = 512 values)
3. Compare corresponding regions between the two images using histogram intersection
4. Combine the 9 region scores using a weighting scheme
5. Sort by final score (ascending) and print the top 5

### Feature extraction (`MultiHistogramFeaturizer`)

Divides the image into a 3×3 grid and computes a normalized 3D histogram per region:

```
grid layout:
+-------+-------+-------+
|  TL   |  TC   |  TR   |
+-------+-------+-------+
|  ML   |  CTR  |  MR   |
+-------+-------+-------+
|  BL   |  BC   |  BR   |
+-------+-------+-------+

bins per channel = 8
cells per region = 8 × 8 × 8 = 512
total features   = 9 × 512 = 4608 values
```

### Distance metrics

Two variants are available:

**`MultiHistogramScoring` (equal weights):**
Each of the 9 regions contributes equally (weight = 1/9).

**`MultiHistogramCenterWeightedScoring` (center-heavy):**
```
corners (TL, TR, BL, BR) → 0.05 each
edges   (TC, ML, MR, BC) → 0.075 each
center  (CTR)            → 0.50
```

### Discussion: equal vs center-weighted

For landscape images where content is spread across the whole frame (e.g. `pic.0274.jpg`), **equal weighting produces better results** — the center-heavy approach over-focuses on one region and misses the global spatial color layout, leading to false matches.

### Run

```bash
# equal weights
./match <target_image> <database_dir> multihistogram

# center-heavy weights
./match <target_image> <database_dir> multihistogram-weighted
```

### Example

```bash
./match ../olympus/pic.0274.jpg ../olympus multihistogram
```

### Example output

```
Top 5 matches for: pic.0274.jpg (method: multihistogram)
1. pic.0273.jpg  (score: -0.571089)
2. pic.1031.jpg  (score: -0.519946)
3. pic.0409.jpg  (score: -0.517759)
4. pic.0275.jpg  (score: -0.49662)
5. pic.0991.jpg  (score: -0.485229)
```
