#!/bin/bash

BINARY=./build/match
DB=./olympus
EMBEDS=./data/ResNet18_olym.csv
NUM_MATCHES=3

TASK=${1:-all}

run_task1() {
    echo "=== Task 1: Baseline (7x7 center patch, SSD) ==="
    $BINARY ./olympus/pic.1016.jpg $DB baseline $NUM_MATCHES
}

run_task2() {
    echo "=== Task 2: Color Histogram (3D BGR, histogram intersection) ==="
    $BINARY ./olympus/pic.0164.jpg $DB histogram $NUM_MATCHES
}

run_task3() {
    echo "=== Task 3: Multi-Histogram (3x3 grid, equal weight) ==="
    $BINARY ./olympus/pic.0274.jpg $DB multihistogram $NUM_MATCHES
    echo ""
    echo "=== Task 3: Multi-Histogram (3x3 grid, center weighted) ==="
    $BINARY ./olympus/pic.0274.jpg $DB multihistogram-weighted $NUM_MATCHES
}

run_task4() {
    echo "=== Task 4: Texture + Color (Sobel gradient mag + 3D color hist) ==="
    $BINARY ./olympus/pic.0535.jpg $DB texture $NUM_MATCHES
}

run_task5() {
    echo "=== Task 5: DNN Embeddings — pic.0893 ==="
    $BINARY ./olympus/pic.0893.jpg $DB dnn $NUM_MATCHES --embeds $EMBEDS
    echo ""
    echo "=== Task 5: DNN Embeddings — pic.0164 ==="
    $BINARY ./olympus/pic.0164.jpg $DB dnn $NUM_MATCHES --embeds $EMBEDS
}

run_task6() {
    echo "=== Task 6: DNN vs Classic — pic.1072 (histogram) ==="
    $BINARY ./olympus/pic.1072.jpg $DB histogram $NUM_MATCHES
    echo ""
    echo "=== Task 6: DNN vs Classic — pic.1072 (dnn) ==="
    $BINARY ./olympus/pic.1072.jpg $DB dnn $NUM_MATCHES --embeds $EMBEDS
    echo ""
    echo "=== Task 6: DNN vs Classic — pic.0948 (histogram) ==="
    $BINARY ./olympus/pic.0948.jpg $DB histogram $NUM_MATCHES
    echo ""
    echo "=== Task 6: DNN vs Classic — pic.0948 (dnn) ==="
    $BINARY ./olympus/pic.0948.jpg $DB dnn $NUM_MATCHES --embeds $EMBEDS
}

run_task7() {
    echo "=== Task 7: Custom — pic.0274 (top $NUM_MATCHES) ==="
    $BINARY ./olympus/pic.0274.jpg $DB custom $NUM_MATCHES --embeds $EMBEDS
    echo ""
    echo "=== Task 7: Custom — pic.0274 (bottom $NUM_MATCHES) ==="
    $BINARY ./olympus/pic.0274.jpg $DB custom $NUM_MATCHES --embeds $EMBEDS bottom
    echo ""
    echo "=== Task 7: Custom — pic.0893 (top $NUM_MATCHES) ==="
    $BINARY ./olympus/pic.0893.jpg $DB custom $NUM_MATCHES --embeds $EMBEDS
    echo ""
    echo "=== Task 7: Custom — pic.0893 (bottom $NUM_MATCHES) ==="
    $BINARY ./olympus/pic.0893.jpg $DB custom $NUM_MATCHES --embeds $EMBEDS bottom
}

case $TASK in
  1) run_task1 ;;
  2) run_task2 ;;
  3) run_task3 ;;
  4) run_task4 ;;
  5) run_task5 ;;
  6) run_task6 ;;
  7) run_task7 ;;
  all)
    run_task1; echo ""
    run_task2; echo ""
    run_task3; echo ""
    run_task4; echo ""
    run_task5; echo ""
    run_task6; echo ""
    run_task7
    ;;
  *) echo "Usage: ./run.sh [1|2|3|4|5|6|7|all]" ;;
esac
