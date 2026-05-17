# DA2 Example Code

This directory contains the Depth Anything V2 wrapper and example programs provided by the course instructor (Bruce A. Maxwell, January 2025).

## Source

Download the original zip from the CS5330 course page on Canvas.

The zip includes:
- `DA2Network.hpp` — ONNX Runtime wrapper for the DA2 network
- `da2-example.cpp` — still image depth estimation example
- `da2-video.cpp` — live video depth estimation example
- `model_fp16.onnx` — the DA2 FP16 model file
- `makefile` — macOS build file (use CMakeLists.txt in this directory for Linux)

## Notes

`DA2Network.hpp` has been modified from the original: the ONNX session is created with
`ORT_ENABLE_BASIC` graph optimization level instead of `nullptr` (default) to fix a
`SimplifiedLayerNormFusion` crash with ONNX Runtime 1.26.0.

## Model file

`model_fp16.onnx` is not checked in (large binary). Download it as part of the zip above
and place it in this directory.
