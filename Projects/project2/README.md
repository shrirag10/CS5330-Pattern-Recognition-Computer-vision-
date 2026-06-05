# Project 2: Content-Based Image Retrieval

## Build

```bash
cmake -S . -B build && cmake --build build
```

---

## Tasks

### Task 1 — Baseline (7x7 center patch, SSD)
```bash
./build/match ./olympus/pic.1016.jpg ./olympus baseline 3
```

### Task 2 — Color Histogram (3D BGR, histogram intersection)
```bash
./build/match ./olympus/pic.0164.jpg ./olympus histogram 3
```

### Task 3 — Multi-Histogram (3x3 spatial grid)
```bash
./build/match ./olympus/pic.0274.jpg ./olympus multihistogram 3
./build/match ./olympus/pic.0274.jpg ./olympus multihistogram-weighted 3
```

### Task 4 — Texture + Color (Sobel gradient mag + 3D color hist)
```bash
./build/match ./olympus/pic.0535.jpg ./olympus texture 3
```

### Task 5 — DNN Embeddings (ResNet18, cosine distance)
```bash
./build/match ./olympus/pic.0893.jpg ./olympus dnn 3 --embeds ./data/ResNet18_olym.csv
./build/match ./olympus/pic.0164.jpg ./olympus dnn 3 --embeds ./data/ResNet18_olym.csv
```

### Task 6 — DNN vs Classic Comparison
```bash
./build/match ./olympus/pic.1072.jpg ./olympus histogram 3
./build/match ./olympus/pic.1072.jpg ./olympus dnn 3 --embeds ./data/ResNet18_olym.csv

./build/match ./olympus/pic.0948.jpg ./olympus histogram 3
./build/match ./olympus/pic.0948.jpg ./olympus dnn 3 --embeds ./data/ResNet18_olym.csv
```

### Task 7 — Custom (color hist + DNN, 40/60 weighted distance)
```bash
./build/match ./olympus/pic.0274.jpg ./olympus custom 5 --embeds ./data/ResNet18_olym.csv
./build/match ./olympus/pic.0274.jpg ./olympus custom 5 --embeds ./data/ResNet18_olym.csv bottom

./build/match ./olympus/pic.0893.jpg ./olympus custom 5 --embeds ./data/ResNet18_olym.csv
./build/match ./olympus/pic.0893.jpg ./olympus custom 5 --embeds ./data/ResNet18_olym.csv bottom
```

---

## Usage

```
./build/match <target_image> <database_dir> <method> [N] [--embeds <csv>] [bottom]
```

- `N` — number of results (default: 5)
- `--embeds` — required for `dnn` and `custom` methods
- `bottom` — show N least similar instead of most similar
