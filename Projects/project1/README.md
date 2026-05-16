# Project 1 - CS5330 Computer Vision

## Build

```bash
mkdir build && cd build

## Run cmake .. only when you make changes to CMakeLists.txt
cmake ..

make
```

## Tasks

| Task | File | Key | Description | Run |
|------|------|-----|-------------|-----|
| 1 | `imgDisplay.cpp` | — | Read an image from a file and display it | `./imgDisplay <image_path>` |
| 2 | `vidDisplay.cpp` | — | Display live video from webcam. Press `s` to save a frame, `q` to quit | `./vidDisplay` |
| 3 | `vidDisplay.cpp` | `g` | Toggle OpenCV cvtColor greyscale | `./vidDisplay` |
| 4 | `vidDisplay.cpp` / `filter.cpp` | `h` | Toggle custom blue-emphasis greyscale | `./vidDisplay` |
| 5 | `vidDisplay.cpp` / `filter.cpp` | `p` | Toggle sepia tone filter | `./vidDisplay` |
| 6 | `vidDisplay.cpp` / `filter.cpp` | `b` / `1` | Toggle fast 5x5 blur (`b`); run naive blur with timing (`1`) | `./vidDisplay` |
| 7 | `vidDisplay.cpp` / `filter.cpp` | `x` / `y` | Toggle Sobel X (`x`) or Sobel Y (`y`) edge detection | `./vidDisplay` |
| 8 | `vidDisplay.cpp` / `filter.cpp` | `m` | Toggle gradient magnitude image | `./vidDisplay` |
| 9 | `vidDisplay.cpp` / `filter.cpp` | `l` | Toggle blur + quantize (10 levels) | `./vidDisplay` |
| 10 | `vidDisplay.cpp` / `faceDetect/faceDetect.cpp` | `f` | Toggle Haar cascade face detection with bounding boxes | `./vidDisplay` |
| 11 | TBD | — | Depth estimation using Depth Anything v2 | — |
| 12 | TBD | — | Custom effects | — |
