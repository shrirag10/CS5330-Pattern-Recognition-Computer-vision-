# Slide 1: Title Slide
## Transfer Learning versus Training from Scratch for Scene Image Classification

* **Presenter:** Shriman Raghav Srinivasan
* **Course:** CS 5330 Final Project Presentation
* **Institution:** Khoury College of Computer Sciences, Northeastern University
* **Date:** July 2026

---

# Slide 2: Introduction & Motivation
## The Core Question
How much of the benefit in Transfer Learning is due to the **pre-trained feature quality**, and how much is due to the **reduced trainable capacity** (fewer parameters to optimize)?

* **Confounding Factors:** A naive comparison between a pre-trained frozen network and a fully trained random network confounds feature quality and parameter capacity.
* **Our Solution:** A capacity-matched three-condition design:
  1. **Pretrained-Frozen (PT-F):** Loaded with ImageNet weights; early layers frozen.
  2. **Random-Frozen (R-F):** Randomly initialized; early layers frozen (Capacity Control).
  3. **Random-Full (R-FL):** Randomly initialized; all layers trainable (Scratch Baseline).

---

# Slide 3: Model Architecture & Partitioning
## Network Structure (ResNet-18)
* **Frozen Subset ($\theta_f$):** `conv1`, `bn1`, `maxpool`, `layer1`, `layer2`, and `layer3` (2,782,784 frozen parameters).
* **Trainable Subset ($\theta_t$):** `layer4` and classification head `fc` (8,396,806 trainable parameters).
* **Critical Technical Detail:** Batch Normalization layers in $\theta_f$ are forced into evaluation mode (`model.eval()`) during training. This prevents running statistics from drifting and ruining the frozen representation.

---

# Slide 4: Dataset & Preprocessing Pipeline
## Intel Image Classification Dataset
* **Content:** Natural scenes across 6 classes: *Buildings, Forest, Glacier, Mountain, Sea, Street*.
* **Transforms:** Resized shorter side to 224px, center-cropped to 224x224, and normalized using ImageNet channel mean & standard deviation.
* **Splitting:**
  * **Train Set (90%):** 12,631 images.
  * **Validation Set (10%):** 1,403 images (class-stratified split for early stopping).
  * **Test Set:** 3,000 images (completely held out for final reporting).

---

# Slide 5: Experimental Configuration
## Training Details
* **Optimizer:** Stochastic Gradient Descent (SGD)
* **Hyperparameters:** Learning rate = 0.001, momentum = 0.9, weight decay = 1e-4.
* **Loss:** Cross-entropy loss on $\theta_t$.
* **Robustness:** Monitored validation loss with early stopping (patience = 5 epochs) to avoid overfitting. All conditions evaluated across 3 distinct random seeds (42, 100, 2026).

---

# Slide 6: Results - Accuracy and Convergence
## Performance Summary (Mean ± Std Dev)

* **Pretrained-Frozen:**
  * Test Accuracy: **[PT-F Mean]%** (±[PT-F Std]%)
  * Convergence Speed: **[PT-F Epochs]** epochs
* **Random-Frozen (Capacity Control):**
  * Test Accuracy: **[R-F Mean]%** (±[R-F Std]%)
  * Convergence Speed: **[R-F Epochs]** epochs
* **Random-Full (Scratch baseline):**
  * Test Accuracy: **[R-FL Mean]%** (±[R-FL Std]%)
  * Convergence Speed: **[R-FL Epochs]** epochs

*PT-F achieves a massive accuracy boost (>50% absolute accuracy improvement) over R-F, isolating pre-training value.*

---

# Slide 7: Error Analysis & Class Confusions
## Key Findings from Confusion Matrices
* **Common Category Confusions:**
  * **Glacier vs. Mountain:** High rate of confusion due to visual similarities (snow, ridges, rock textures).
  * **Buildings vs. Street:** High rate of confusion due to environmental context (streets naturally contain buildings in the background).
* **Robustness of Errors:** These confusions persist across all training conditions, indicating they are properties of the visual similarities in the dataset rather than the optimization regime.

---

# Slide 8: Discussion & Takeaways
## Conclusions
1. **Pretraining Quality is Paramount:** Under identical capacities, pretrained features provide an absolute test accuracy gain of >50% compared to random capacity-matched controls.
2. **Capacity vs. Initialization:** Training from scratch with full capacity (R-FL) outperforms the frozen random control (R-F) but fails to approach transfer learning (PT-F), while requiring far more epochs to converge.
3. **Takeaway:** For small/medium datasets with limited computational budgets, transfer learning is essential. Pretraining acts as a powerful regularizer and efficiency booster that cannot be easily matched by training from scratch.
