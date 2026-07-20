# CS 5330 Final Project: Transfer Learning versus Training from Scratch for Scene Image Classification

* **Author:** Shriman Raghav Srinivasan (Individual Project)
* **Email:** srinivasan.shri@northeastern.edu
* **Northeastern University, Khoury College of Computer Sciences**

---

## Project Description

This project performs a rigorous, capacity-matched empirical comparison between **Transfer Learning** and **Training from Scratch** using a ResNet-18 backbone on the six-class **Intel Image Classification** dataset. 

A standard comparison between a pre-trained network with frozen early layers and a fully trained network from scratch confounds two effects: the quality of the pretrained representations, and the reduced trainable capacity of the frozen network. To isolate these factors, we implement a three-condition design:
1. **Pretrained-Frozen (PT-F):** Loaded with ImageNet-pretrained weights, early layers ($\theta_f$) are frozen, and only the final convolutional block (`layer4`) and linear head ($\theta_t$) are trainable.
2. **Random-Frozen (R-F):** Capacity-matched control identical to PT-F, but early layers ($\theta_f$) are randomly initialized and frozen.
3. **Random-Full (R-FL):** Fully trained from scratch (all parameters trainable).

In all frozen conditions, Batch Normalization layers within $\theta_f$ are forced into evaluation mode (`model.eval()`) during training to prevent representation drift. All conditions are optimized using SGD with momentum=0.9, weight decay=1e-4, and learning rate 1e-3, across 3 distinct random seeds (42, 100, 2026), terminated via validation-based early stopping (patience=5).

---

## Deliverables & Paths

* **LaTeX Code & PDF Report:** [report/report.tex](report/report.tex), compiled to [report/report.pdf](report/report.pdf)
* **Presentation Slides PDF:** [presentation/presentation.pdf](presentation/presentation.pdf) (derived from Beamer source [presentation/presentation.tex](presentation/presentation.tex))
* **Trained Models / Checkpoints:** Saved under `results/` as `best_model_{condition}_seed{seed}.pth`
* **Plot Figures & Metrics:** Saved under `results/` (`loss_curves.png`, `accuracy_curves.png`, `confusion_matrices.png`, `summary_metrics.json`, `test_evaluation_metrics.json`)

---

## Dataset & Presentation Links

* **Dataset Source URL (Kaggle):** [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
* **Dataset Direct Zip Download Mirror:** [https://huggingface.co/datasets/miladfa7/Intel-Image-Classification](https://huggingface.co/datasets/miladfa7/Intel-Image-Classification)
* **Demo/Presentation Video URL:** [https://drive.google.com/drive/folders/placeholder-url-for-presentation-video](https://drive.google.com/drive/folders/placeholder-url-for-presentation-video) (or upload to YouTube)

---

## Code Directory Structure

```
final_project/
├── data/                      # Dataset files (seg_train/ and seg_test/)
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Preprocessing transforms and stratified validation splits
│   ├── model.py               # ResNet-18 model setups and parameter freezing
│   ├── train.py               # Training loop with early stopping logic
│   ├── evaluate.py            # Evaluation on test split, precision/recall/F1 calculations
│   ├── run_experiments.py     # Grid runner script (runs 9 jobs)
│   ├── visualize.py           # Plots training curves and confusion matrix heatmaps
│   └── generate_latex_results.py # Outputs results_summary.tex macros from JSON summaries
├── results/                   # Checkpoints, metric JSON files, and image plots
├── report/                    # LaTeX source and compiled report
├── presentation/              # Beamer slide deck presentation
├── main.py                    # Entry point script
└── verify_pipeline.py         # Mock pipeline testing utility
```

---

## Setup and Running Instructions

### 1. Requirements
Ensure you have PyTorch, torchvision, numpy, and matplotlib installed:
```bash
pip install torch torchvision numpy matplotlib
```

### 2. Dataset Acquisition
The dataset is automatically downloaded and extracted into the `data/` folder during execution, or you can manually place the `seg_train/`, `seg_test/`, and `seg_pred/` folders under `data/`.

### 3. Verification Check
Run the quick integration test (takes ~5 seconds) to verify model partitioning, batch norm handling, and check that your environment runs smoothly:
```bash
python3 verify_pipeline.py
```

### 4. Running the Full Experiment Grid
To run the full 9 training experiments (3 conditions $\times$ 3 seeds) with early stopping, execute:
```bash
python3 main.py --run
```
All checkpoints and training curve data will be saved under `results/`.

### 5. Running Evaluation & Generating Plots
To run evaluation on the held-out test split and create the confusion matrices and learning curves, run:
```bash
python3 main.py --eval
```
This updates the summary files `results/summary_metrics.json` and `results/test_evaluation_metrics.json`, and outputs the plots `loss_curves.png`, `accuracy_curves.png`, and `confusion_matrices.png`.

### 6. Compiling the LaTeX Report & Slides
To generate the automated results summary macro file and compile the report and presentation:
```bash
python3 main.py --compile
```
This generates `report/results_summary.tex` and runs `pdflatex` to compile `report/report.pdf` and `presentation/presentation.pdf` automatically with the correct metrics.
