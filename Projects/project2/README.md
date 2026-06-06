# Project 2: Content-Based Image Retrieval

**Authors:** Shriman Raghav Srinivasan, Shyam Sreenivasan
**Course:** CS5330 Computer Vision — Northeastern University

---

## Environment

- **OS:** Ubuntu 22.04 / Linux
- **IDE:** VS Code with clangd
- **Compiler:** g++ (C++17), CMake 3.10+
- **Dependencies:** OpenCV 4.x, Python 3 with PyQt5 and matplotlib

---

## Build

```bash
cmake -S . -B build && cmake --build build
```

## Run All Tasks

```bash
./run.sh all       # run all tasks
./run.sh 1         # run a specific task (1-7)
```

---

## Running the Matcher

```
./build/match <target_image> <database_dir> <method> [N] [--embeds <csv>] [bottom]
```

| Argument | Description |
|---|---|
| `<target_image>` | Path to target image |
| `<database_dir>` | Directory of database images (e.g. `./olympus`) |
| `<method>` | See methods table below |
| `N` | Number of results (default: 5) |
| `--embeds <csv>` | Required for `dnn` and `custom` methods |
| `bottom` | Show N least similar instead of most similar |

### Methods

| Method | Feature | Distance |
|---|---|---|
| `baseline` | 7×7 center patch (147 values) | SSD |
| `histogram` | 3D BGR histogram (16 bins/ch, 4096 values) | Histogram intersection |
| `multihistogram` | 3×3 spatial grid histograms (4608 values) | Histogram intersection (equal weight) |
| `multihistogram-weighted` | Same as above | Center-weighted intersection |
| `texture` | Color hist + Sobel gradient mag hist (528 values) | Equal-weight intersection |
| `dnn` | ResNet18 embeddings from CSV (512 values) | Cosine distance |
| `custom` | Color hist + DNN (1024 values) | 40% color + 60% cosine |

---

## Task Examples

### Task 1 — Baseline
```bash
./build/match ./olympus/pic.1016.jpg ./olympus baseline 3
```

### Task 2 — Histogram
```bash
./build/match ./olympus/pic.0164.jpg ./olympus histogram 3
```

### Task 3 — Multi-Histogram
```bash
./build/match ./olympus/pic.0274.jpg ./olympus multihistogram 3
./build/match ./olympus/pic.0274.jpg ./olympus multihistogram-weighted 3
```

### Task 4 — Texture + Color
```bash
./build/match ./olympus/pic.0535.jpg ./olympus texture 3
```

### Task 5 — DNN Embeddings
```bash
./build/match ./olympus/pic.0893.jpg ./olympus dnn 3 --embeds ./ResNet18_olym.csv
./build/match ./olympus/pic.0164.jpg ./olympus dnn 3 --embeds ./ResNet18_olym.csv
```

### Task 6 — DNN vs Classic Comparison
```bash
./build/match ./olympus/pic.1072.jpg ./olympus histogram 3
./build/match ./olympus/pic.1072.jpg ./olympus dnn 3 --embeds ./ResNet18_olym.csv

./build/match ./olympus/pic.0948.jpg ./olympus histogram 3
./build/match ./olympus/pic.0948.jpg ./olympus dnn 3 --embeds ./ResNet18_olym.csv
```

### Task 7 — Custom Method (top 5 + bottom 5)
```bash
./build/match ./olympus/pic.0274.jpg ./olympus custom 5 --embeds ./ResNet18_olym.csv
./build/match ./olympus/pic.0274.jpg ./olympus custom 5 --embeds ./ResNet18_olym.csv bottom

./build/match ./olympus/pic.0893.jpg ./olympus custom 5 --embeds ./ResNet18_olym.csv
./build/match ./olympus/pic.0893.jpg ./olympus custom 5 --embeds ./ResNet18_olym.csv bottom
```

---

## Extensions

### Extension 1: Center-Weighted Spatial Histogram
```bash
./build/match ./olympus/pic.0274.jpg ./olympus multihistogram-weighted 3
```

### Extension 2: Python Visual Query Viewer
Requires: `pip install matplotlib`
```bash
python3 show_matches.py ./olympus/pic.1016.jpg ./olympus baseline 3
python3 show_matches.py ./olympus/pic.0164.jpg ./olympus dnn 5 --embeds ./ResNet18_olym.csv
python3 show_matches.py ./olympus/pic.0274.jpg ./olympus custom 5 --embeds ./ResNet18_olym.csv
```

### Extension 3: Interactive PyQt5 GUI
Requires: `pip install PyQt5`
```bash
python3 gui.py
```
Browse for a target image, select the database directory (default: `olympus/`), choose a method, set N, and click **Find Matches**. For `dnn` and `custom` methods, the GUI automatically locates `ResNet18_olym.csv` one level above the selected database folder.

---

## Time Travel Days

Not using any time travel days.
