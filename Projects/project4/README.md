# CS5330 Project 4: Camera Calibration and Augmented Reality

**Author:** Shriman Raghav Srinivasan (Single Person Project)
**Date:** June 2026

## Project Overview
This project is about camera calibration and augmented reality. I will build a system to detect a target pattern (a 9x6 chessboard), calibrate the camera to find intrinsic parameters, and project 3D virtual objects onto the target in real time.

## Directory Structure
- `CMakeLists.txt`: Build configuration file.
- `calibrate.cpp`: Calibration application.
- `ar_system.cpp`: Augmented reality projection application.
- `robust_features.cpp`: Robust features display application.
- `generate_checkerboard.py`: Helper script to generate a 10x7 checkerboard image (9x6 internal corners).

## Build Instructions
To build the binaries, run the following commands:
```bash
mkdir -p build
cd build
cmake ..
make
```

## Running the Applications
1. **Target Generation**:
   ```bash
   python3 generate_checkerboard.py
   ```
2. **Calibration**:
   ```bash
   ./calibrate
   ```
   - Press `s` to save a calibration frame when the target is detected.
   - Press `c` once 5+ frames are saved to compute camera parameters.
   - Press `w` to write parameters to `calibration_params.xml`.
3. **AR Overlay**:
   ```bash
   ./ar_system
   ```
4. **Robust Features**:
   ```bash
   ./robust_features
   ```
