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
├── SimilarityScoringImpl.h   # SSDScoring implementation
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
./match <target_image> <database_dir>
```

### Example

```bash
./match ../olympus/pic.1016.jpg ../olympus
```

### Example output

```
Top 5 matches for: pic.1016.jpg
1. pic.0986.jpg  (SSD: ...)
2. pic.0641.jpg  (SSD: ...)
3. pic.0547.jpg  (SSD: ...)
4. pic.1013.jpg  (SSD: ...)
5. ...
```
