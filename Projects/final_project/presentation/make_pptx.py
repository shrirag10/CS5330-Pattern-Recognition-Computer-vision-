"""
Generate a professional, highly detailed academic .pptx presentation.
Matches the user's reference style:
- Clean white backgrounds
- Georgia Serif for headers/numbers, Calibri for body
- Crimson/Burgundy accent color (#991B1B)
- Slate/Charcoal text color (#1E293B)
- Clean horizontal divider lines
- 7 slides total (less than 8 slides)
- Highly dense, informative, and complete content
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

ASSETS = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation/assets"
OUT = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation/presentation.pptx"

# Color Palette
CRIMSON = RGBColor(153, 27, 27)    # Burgundy accent (#991B1B)
SLATE_900 = RGBColor(15, 23, 42)    # Slate dark text (#0F172A)
SLATE_700 = RGBColor(51, 65, 85)    # Body text (#334155)
SLATE_400 = RGBColor(148, 163, 184) # Muted text (#94A3B8)
LIGHT_GREY = RGBColor(226, 232, 240) # Divider lines (#E2E8F0)
WHITE = RGBColor(255, 255, 255)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# --- General Helpers ---

def set_slide_bg_white(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = WHITE

def add_slide_header(slide, number_str, title_str):
    txBox = slide.shapes.add_textbox(Inches(0.7), Inches(0.4), Inches(11.933), Inches(0.8))
    tf = txBox.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    
    # Add slide number in Georgia Crimson
    run_num = p.add_run()
    run_num.text = number_str + "    "
    run_num.font.name = 'Georgia'
    run_num.font.size = Pt(24)
    run_num.font.bold = True
    run_num.font.color.rgb = CRIMSON
    
    # Add title in Georgia Slate
    run_title = p.add_run()
    run_title.text = title_str
    run_title.font.name = 'Georgia'
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = SLATE_900
    
    # Add thin divider line below header
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.1), Inches(11.933), Pt(1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_GREY
    shape.line.fill.background()

def add_slide_number(slide, num, total=7):
    add_textbox(slide, Inches(11.633), Inches(6.8), Inches(1), Inches(0.3),
                f"{num} / {total}", size=11, bold=False, color=SLATE_400, align=PP_ALIGN.RIGHT)

def add_textbox(slide, left, top, width, height, text, size=14, bold=False, color=SLATE_700, align=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return txBox

def add_paragraph_to_frame(tf, text, size=14, bold=False, color=SLATE_700, space_before=Pt(8)):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = 'Calibri'
    p.space_before = space_before
    return p

def add_numbered_item(tf, num_str, title_str, desc_str, bold_desc=False):
    p = tf.add_paragraph()
    p.space_before = Pt(14)
    p.space_after = Pt(2)
    
    # Number
    run_num = p.add_run()
    run_num.text = num_str + "    "
    run_num.font.name = 'Georgia'
    run_num.font.size = Pt(14)
    run_num.font.bold = True
    run_num.font.color.rgb = CRIMSON
    
    # Bold Title
    run_title = p.add_run()
    run_title.text = title_str
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(14)
    run_title.font.bold = True
    run_title.font.color.rgb = SLATE_900
    
    # Description
    run_desc = p.add_run()
    run_desc.text = " — " + desc_str
    run_desc.font.name = 'Calibri'
    run_desc.font.size = Pt(13)
    run_desc.font.bold = bold_desc
    run_desc.font.color.rgb = SLATE_700

# ═══════════════════════════════════════════
# SLIDE 1: TITLE SLIDE
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)

# Subtitle / Metadata
add_textbox(slide, Inches(0.7), Inches(1.8), Inches(11.9), Inches(0.4),
            "CS 5330 — Computer Vision — Final Project", size=13, bold=False, color=SLATE_400, font_name='Calibri')

# Title
add_textbox(slide, Inches(0.7), Inches(2.2), Inches(11.9), Inches(1.4),
            "Transfer Learning vs.\nTraining from Scratch", size=38, bold=True, color=SLATE_900, font_name='Georgia')

# Burgundy accent line below title
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(3.8), Inches(2.8), Pt(3))
shape.fill.solid()
shape.fill.fore_color.rgb = CRIMSON
shape.line.fill.background()

# Author & Info
add_textbox(slide, Inches(0.7), Inches(4.2), Inches(11.9), Inches(0.4),
            "A Capacity-Matched Empirical Evaluation using ResNet-18 on the Intel Image Dataset", size=15, bold=False, color=SLATE_700, font_name='Calibri')

add_textbox(slide, Inches(0.7), Inches(5.2), Inches(11.9), Inches(1.2),
            "Author: Shriman Raghav Srinivasan\nNortheastern University  ·  Khoury College of Computer Sciences\nAdvisors: CS 5330 Computer Vision Instruction Team",
            size=13, bold=False, color=SLATE_700, font_name='Calibri')

add_slide_number(slide, 1)


# ═══════════════════════════════════════════
# SLIDE 2: 01 THE CORE CONFOUND & EXPERIMENTAL DESIGN
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "01", "The Confound & Experimental Design")

# Left Column: Motivation and Mathematical formulation
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.8),
                      "Motivation & Formulation", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "The Core Confound in Transfer Comparisons", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "Standard evaluations contrast a partially frozen, pre-trained network with a fully trained network from scratch. This confounds weight initialization quality with model trainable capacity (parameter count).", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Mathematical Optimization Objective", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "Let the network parameters θ be split into frozen θf and trainable θt. For all conditions, we optimize θt to minimize cross-entropy loss over N samples:", size=13, color=SLATE_700, space_before=Pt(2))

p_math = tf_left.add_paragraph()
p_math.text = "   L(θt) = – (1/N) * Σ [ log p_θ(yi | xi) ]"
p_math.font.name = 'Georgia'
p_math.font.size = Pt(13)
p_math.font.bold = True
p_math.font.color.rgb = CRIMSON
p_math.space_before = Pt(8)
p_math.space_after = Pt(8)

add_paragraph_to_frame(tf_left, "Isolating Feature Quality", size=13, bold=True, color=SLATE_900, space_before=Pt(8))
add_paragraph_to_frame(tf_left, "We hold trainable capacity constant and vary weight initialization to measure the exact contribution of pre-training features.", size=12, color=SLATE_700, space_before=Pt(2))

# Right Column: The 3 Conditions
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.8),
                       "Three Training Conditions", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_numbered_item(tf_right, "1", "Pretrained-Frozen (PT-F)", "Initialized with ImageNet weights. Early layers (θf) frozen; only the final residual stage (layer4) and classification head (fc) are trainable (θt).")
add_numbered_item(tf_right, "2", "Random-Frozen (R-F)", "Early layers (θf) randomly initialized and frozen. Uses the exact same trainable capacity (θt) as PT-F. Acts as the capacity-matched control.")
add_numbered_item(tf_right, "3", "Random-Full (R-FL)", "Fully trained from scratch. All parameters (both θf and θt) are trainable from random initialization. Serves as the classic capacity baseline.")

add_slide_number(slide, 2)


# ═══════════════════════════════════════════
# SLIDE 3: 02 DATASET DETAILS & ARCHITECTURE PARTITIONING
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "02", "Dataset & ResNet-18 Partitioning")

# Left Column: Dataset & Preprocessing Details
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.8),
                      "Intel Dataset & Preprocessing", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Data Distribution & Splits", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "Intel Image Classification dataset contains 150×150 px natural scene images across 6 classes: buildings, forest, glacier, mountain, sea, street. Standard training split (14,034 images) is divided into 12,631 train (90%) and 1,403 validation (10%) via class-stratified sampling. Held-out test set contains 3,000 images.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Strict Preprocessing Pipeline", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "To match the model architecture requirements and maintain domain consistency:\n• Resize shorter side to 224px (bilinear interpolation)\n• Center-crop to 224 × 224 pixels\n• Per-channel ImageNet normalize: μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]", size=13, color=SLATE_700, space_before=Pt(2))

# Right Column: Architecture & BN Handling
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.8),
                       "ResNet-18 Partition Boundaries", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_numbered_item(tf_right, "1", "Frozen Parameters Block (θf)", "Input convolution (conv1), first batchnorm (bn1), and residual stages layer1, layer2, and layer3. Total parameter count is 2,782,784 (frozen).")
add_numbered_item(tf_right, "2", "Trainable Parameters Block (θt)", "Final residual stage (layer4) and 6-class output fully-connected layer (fc). Total parameter count is 8,396,806 (optimized).")
add_numbered_item(tf_right, "3", "Locked Batch Normalization Rule", "BN layers in θf are forced into eval() mode during training. This locks their running mean and variance, preventing update updates from corrupting representation statistics.")

add_slide_number(slide, 3)


# ═══════════════════════════════════════════
# SLIDE 4: 03 PERFORMANCE EVALUATION & STATISTICS
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "03", "Test Performance & Statistics")

# Left Column: Accuracy Metrics
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.8),
                      "Experimental Metrics", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Empirical Test Results (Mean ± Std Dev)", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "Evaluated on the held-out test split of 3,000 images across 3 random seeds (42, 100, 2026):", size=13, color=SLATE_700, space_before=Pt(2))

# Table inside left column
tbl_shape = slide.shapes.add_table(4, 3, Inches(0.7), Inches(2.7), Inches(5.5), Inches(1.4))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(2.3)
tbl.columns[1].width = Inches(1.6)
tbl.columns[2].width = Inches(1.6)

headers = ["Condition", "Mean Test Acc", "Std Dev"]
for c, h in enumerate(headers):
    cell = tbl.cell(0, c)
    cell.text = h
    cell.text_frame.paragraphs[0].font.size = Pt(11)
    cell.text_frame.paragraphs[0].font.bold = True
    cell.text_frame.paragraphs[0].font.color.rgb = SLATE_900

rows_data = [
    ["Pretrained-Frozen", "93.27%", "±0.23%"],
    ["Random-Full", "84.69%", "±1.87%"],
    ["Random-Frozen", "66.54%", "±4.04%"]
]
for r, row in enumerate(rows_data):
    for c, val in enumerate(row):
        cell = tbl.cell(r + 1, c)
        cell.text = val
        cell.text_frame.paragraphs[0].font.size = Pt(11)
        if c == 1:
            cell.text_frame.paragraphs[0].font.bold = True
            cell.text_frame.paragraphs[0].font.color.rgb = CRIMSON if r == 0 else SLATE_900

add_paragraph_to_frame(tf_left, "Key Statistical Takeaway", size=13, bold=True, color=SLATE_900, space_before=Pt(1.2 * 72))
add_paragraph_to_frame(tf_left, "PT-F outperforms R-F by 26.73% absolute accuracy, isolating the vast superiority of ImageNet weights under identical model capacity. PT-F also beats R-FL (fully trainable scratch baseline) by 8.58%, illustrating that pre-training functions as a stronger regularizer than network scale alone.", size=12, color=SLATE_700, space_before=Pt(2))

# Right Column: Bar Chart
slide.shapes.add_picture(os.path.join(ASSETS, "bar.png"),
                         Inches(6.8), Inches(1.6), Inches(5.8))

add_slide_number(slide, 4)


# ═══════════════════════════════════════════
# SLIDE 5: 04 TRAINING CURVES & OPTIMIZATION PARAMETERS
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "04", "Training Dynamics & Optimization")

# Left Column: Curves
slide.shapes.add_picture(os.path.join(ASSETS, "curves.png"),
                         Inches(0.7), Inches(1.6), Inches(6.8))

# Right Column: Convergence analysis
tb_right = add_textbox(slide, Inches(8.0), Inches(1.6), Inches(4.6), Inches(4.8),
                       "Optimization & Speed", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_paragraph_to_frame(tf_right, "Training Hyperparameters", size=13, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_right, "• Optimizer: SGD with momentum = 0.9\n• Weight Decay: 1 × 10⁻⁴ | Learning Rate: 1 × 10⁻³\n• Batch Size: 32 | Validation patience = 5 epochs", size=12, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Convergence Epochs & Compute Duration", size=13, bold=True, color=SLATE_900, space_before=Pt(12))
add_paragraph_to_frame(tf_right, "• PT-Frozen: Converges in 8.0 epochs mean (102.5s duration). Starts immediately at high val accuracy (>90% in epoch 1).\n• Random-Frozen: Converges in 8.7 epochs mean (112.8s duration) but plateaus at low validation accuracy (67.8%).\n• Random-Full: Requires 15.0 epochs mean (425.7s duration). Needs nearly twice as many epochs and 4× more compute time.", size=12, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Early Stopping Behavior", size=13, bold=True, color=SLATE_900, space_before=Pt(12))
add_paragraph_to_frame(tf_right, "Early stopping on validation loss successfully prevents overfitting. R-FL continues to improve steadily but exhibits high seed variance, whereas PT-F converges to a flat validation trajectory immediately.", size=12, color=SLATE_700, space_before=Pt(2))

add_slide_number(slide, 5)


# ═══════════════════════════════════════════
# SLIDE 6: 05 CLASS CONFUSIONS & F1 ANALYSIS
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "05", "Confusion & Class F1 Analysis")

# Left Column: Confusion Matrices
slide.shapes.add_picture(os.path.join(ASSETS, "confusion.png"),
                         Inches(0.7), Inches(1.6), Inches(6.8))

# Right Column: Detailed Observations
tb_right = add_textbox(slide, Inches(8.0), Inches(1.6), Inches(4.6), Inches(4.8),
                       "Per-class Findings", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_paragraph_to_frame(tf_right, "Glacier vs. Mountain Confusion", size=13, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_right, "Glaciers and mountains exhibit high visual similarity (white snow, jagged grey rock textures), causing major mutual confusion across all conditions. Representative PT-F seed misclassifies 41 mountain images as glacier and 55 glacier images as mountain.", size=12, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Buildings vs. Street Confusion", size=13, bold=True, color=SLATE_900, space_before=Pt(12))
add_paragraph_to_frame(tf_right, "Urban images frequently co-occur (streets contain buildings in the background), leading to high confusion rates (29 street images classified as buildings).", size=12, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Strongest Categories", size=13, bold=True, color=SLATE_900, space_before=Pt(12))
add_paragraph_to_frame(tf_right, "Forest has the highest F1 score (PT-F: 99.8%) due to distinct green hue and texture patterns. Sea achieves excellent F1 (96.5%).", size=12, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Scratch training F1 drop", size=13, bold=True, color=SLATE_900, space_before=Pt(12))
add_paragraph_to_frame(tf_right, "R-FL achieves lower F1 scores across the board (Forest: 94.8%, Glacier: 73.5%, Mountain: 78.3%), proving that scratch features struggle with subtle boundary details.", size=12, color=SLATE_700, space_before=Pt(2))

add_slide_number(slide, 6)


# ═══════════════════════════════════════════
# SLIDE 7: 06 SCIENTIFIC CONCLUSIONS & DISCUSSION
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "06", "Discussion & Conclusions")

# Left Column: Key Findings
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.8),
                      "Scientific Discussion", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Pretraining Quality vs. Trainable Capacity", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "The capacity control (R-F) establishes that freezing early layers restricts trainable capacity, leading to poor scratch performance (+26.73% accuracy gap). In contrast, pre-trained weights (PT-F) furnish structured, highly generalizable features (edges, textures, shapes) that adapt seamlessly.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Revisiting Literature (He et al. 2019)", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "He et al. showed that training from scratch can match pre-training with enough time and data. Our results confirm that under a tight training patience budget and modest target data size, transfer learning is highly essential, outperforming scratch by 8.58% in 4× less time.", size=13, color=SLATE_700, space_before=Pt(2))

# Right Column: Future Directions
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.8),
                       "Future Work & Summary", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_numbered_item(tf_right, "1", "Vary the frozen boundary", "Test fine-tuning with fewer/more frozen layers (e.g. freeze layer1-2) to investigate where task-specific representations begin.", bold_desc=False)
add_numbered_item(tf_right, "2", "Evaluate larger backbones", "Extend this capacity-matched analysis to deeper models (ResNet-50) or non-convolutional backbones (Vision Transformers).", bold_desc=False)
add_numbered_item(tf_right, "3", "Progressive unfreezing", "Apply learning rate warmup and gradually unlock layers from back to front during training to avoid representation collapse.", bold_desc=False)
add_numbered_item(tf_right, "4", "Advanced augmentations", "Examine if severe augmentations (MixUp, RandAugment) can help scratch models (R-FL) close the generalization gap.", bold_desc=False)

add_slide_number(slide, 7)

# Save
prs.save(OUT)
print(f"✅ Generated minimal professional presentation matching reference style at: {OUT}")
