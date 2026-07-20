"""
Generate a professional, minimal academic .pptx presentation.
Matches the user's reference style:
- Clean white backgrounds
- Georgia Serif for headers, Calibri for body
- Crimson/Burgundy accent color (#991B1B)
- Slate/Charcoal text color (#1E293B)
- Clean horizontal divider lines
- 7 slides total (less than 8 slides)
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
CRIMSON = RGBColor(153, 27, 27)    # Burgundy/Red accent (#991B1B)
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

def add_numbered_item(tf, num_str, title_str, desc_str):
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
    run_desc.font.color.rgb = SLATE_700

# ═══════════════════════════════════════════
# SLIDE 1: TITLE SLIDE
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)

# Subtitle / Metadata
add_textbox(slide, Inches(0.7), Inches(2.2), Inches(11.9), Inches(0.4),
            "CS 5330 — Computer Vision — Final Project", size=13, bold=False, color=SLATE_400, font_name='Calibri')

# Title
add_textbox(slide, Inches(0.7), Inches(2.7), Inches(11.9), Inches(1.4),
            "Key Insights on Transfer Learning\nvs. Training from Scratch", size=38, bold=True, color=SLATE_900, font_name='Georgia')

# Burgundy accent line below title
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(4.3), Inches(2.8), Pt(3))
shape.fill.solid()
shape.fill.fore_color.rgb = CRIMSON
shape.line.fill.background()

# Author & Info
add_textbox(slide, Inches(0.7), Inches(4.7), Inches(11.9), Inches(0.4),
            "A Capacity-Matched Empirical Evaluation using ResNet-18", size=15, bold=False, color=SLATE_700, font_name='Calibri')

add_textbox(slide, Inches(0.7), Inches(5.9), Inches(11.9), Inches(0.4),
            "Shriman Raghav Srinivasan", size=14, bold=True, color=SLATE_900, font_name='Calibri')

add_slide_number(slide, 1)


# ═══════════════════════════════════════════
# SLIDE 2: 01 WHERE THIS SITS (THE PROBLEM)
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "01", "Where this sits")

# Left Column: The Core Ambiguity
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.5),
                      "You already have", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Transfer learning advantage.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "It consistently outperforms scratch training, but naively comparing a partially frozen model to a full model changes two variables at once.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Trainable capacity confound.", size=14, bold=True, color=SLATE_900, space_before=Pt(18))
add_paragraph_to_frame(tf_left, "Does the improvement stem from the high quality of pretrained weights, or is it due to having a restricted subset of parameters to optimize?", size=13, color=SLATE_700, space_before=Pt(2))

# Right Column: The Question & 3 Conditions
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.5),
                       "The question for today", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

p_q = tf_right.add_paragraph()
p_q.text = "How do we isolate pretrained feature quality?"
p_q.font.size = Pt(16)
p_q.font.bold = True
p_q.font.italic = True
p_q.font.color.rgb = CRIMSON
p_q.space_before = Pt(8)

add_numbered_item(tf_right, "1", "Pretrained-Frozen (PT-F)", "Early layers frozen with ImageNet weights. Only layer4 + fc learn.")
add_numbered_item(tf_right, "2", "Random-Frozen (R-F)", "Early layers frozen with random weights. Identical capacity to PT-F.")
add_numbered_item(tf_right, "3", "Random-Full (R-FL)", "All layers trainable from random init. Standard scratch training baseline.")

add_slide_number(slide, 2)


# ═══════════════════════════════════════════
# SLIDE 3: 02 DATA & ARCHITECTURE
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "02", "Dataset & Partitioning")

# Left Column: Dataset & Preprocessing
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.5),
                      "The dataset as a unit", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Intel Image Classification dataset.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "6 scene categories: buildings, forest, glacier, mountain, sea, and street.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Standard splits.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "14,034 training images (carved into 90% train, 10% stratified val) and 3,000 held-out test images.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Identical preprocessing.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "Images are resized to 224px, center cropped to 224×224, and normalized using ImageNet channel mean and std.", size=13, color=SLATE_700, space_before=Pt(2))

# Right Column: Architecture partitioning & BN
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.5),
                       "Network partitioning", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_numbered_item(tf_right, "1", "Frozen Subset (θf)", "conv1, bn1, layer1, layer2, layer3 (2.78M params).")
add_numbered_item(tf_right, "2", "Trainable Subset (θt)", "layer4 and classification head fc (8.40M params).")
add_numbered_item(tf_right, "3", "Batch Normalization handling", "BN layers in θf are explicitly locked in eval() mode. Crucial to prevent statistics drift when weights are frozen.")

add_slide_number(slide, 3)


# ═══════════════════════════════════════════
# SLIDE 4: 03 ACCURACY COMPARISON
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "03", "Test Accuracy Results")

# Left Column: Accuracy Metrics
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.5),
                      "Performance summary", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Pretrained-Frozen (PT-F) is the clear winner.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "Achieves a mean test accuracy of 93.27% ± 0.23% across all random seeds.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Random-Full (R-FL) scratch baseline.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "Achieves a mean test accuracy of 84.69% ± 1.87% when all layers are trained.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Random-Frozen (R-F) capacity control.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "Achieves only 66.54% ± 4.04%, stagnating early due to arbitrary frozen feature space.", size=13, color=SLATE_700, space_before=Pt(2))

p_gap = tf_left.add_paragraph()
p_gap.text = "Matched capacity gap: +26.73% accuracy purely due to pretrained weight quality (PT-F vs. R-F)."
p_gap.font.size = Pt(12)
p_gap.font.bold = True
p_gap.font.color.rgb = CRIMSON
p_gap.space_before = Pt(14)

# Right Column: Bar Chart
slide.shapes.add_picture(os.path.join(ASSETS, "bar.png"),
                         Inches(6.8), Inches(1.6), Inches(5.8))

add_slide_number(slide, 4)


# ═══════════════════════════════════════════
# SLIDE 5: 04 TRAINING CURVES & CONVERGENCE
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "04", "Training Dynamics")

# Left Column: Curves
slide.shapes.add_picture(os.path.join(ASSETS, "curves.png"),
                         Inches(0.7), Inches(1.6), Inches(6.8))

# Right Column: Convergence analysis
tb_right = add_textbox(slide, Inches(8.0), Inches(1.6), Inches(4.6), Inches(4.5),
                       "Convergence & speed", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_paragraph_to_frame(tf_right, "Pretraining yields instant start.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_right, "PT-F begins above 90% validation accuracy at epoch 1 and converges rapidly (mean: 8.0 epochs / 1.7 minutes).", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Scratch is slow and expensive.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_right, "R-FL requires a mean of 15.0 epochs / 7.1 minutes to converge — twice as many epochs and 4× more training time.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Random-frozen bottleneck.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_right, "R-F plateaus quickly (mean: 8.7 epochs / 1.9 minutes) because learning cannot progress beyond arbitrary linear layers.", size=13, color=SLATE_700, space_before=Pt(2))

add_slide_number(slide, 5)


# ═══════════════════════════════════════════
# SLIDE 6: 05 CLASS CONFUSIONS & F1
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "05", "Confusion & Class F1")

# Left Column: Confusion Matrices
slide.shapes.add_picture(os.path.join(ASSETS, "confusion.png"),
                         Inches(0.7), Inches(1.6), Inches(6.8))

# Right Column: Detailed Observations
tb_right = add_textbox(slide, Inches(8.0), Inches(1.6), Inches(4.6), Inches(4.5),
                       "Per-class findings", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_paragraph_to_frame(tf_right, "Glacier vs. Mountain confusion.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_right, "Shared visual signatures (snow, rock textures) lead to major mutual confusion in all conditions.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Buildings vs. Street confusion.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_right, "Urban context overlap makes these classes prone to mutual misclassifications.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_right, "Distinct classes excel.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_right, "Forest consistently achieves the highest F1 score due to strong, uniform color and texture cues.", size=13, color=SLATE_700, space_before=Pt(2))

add_slide_number(slide, 6)


# ═══════════════════════════════════════════
# SLIDE 7: 06 TAKEAWAYS & CONCLUSIONS
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg_white(slide)
add_slide_header(slide, "06", "Key Takeaways")

# Left Column: Key Findings
tb_left = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.5), Inches(4.5),
                      "Conclusions", size=15, bold=True, color=SLATE_400)
tf_left = tb_left.text_frame

add_paragraph_to_frame(tf_left, "Pretrained features drive transfer success.", size=14, bold=True, color=SLATE_900)
add_paragraph_to_frame(tf_left, "The capacity control (R-F) confirms that restricting trainable parameter space is not what makes transfer learning succeed. Pretrained feature quality is the primary driver.", size=13, color=SLATE_700, space_before=Pt(2))

add_paragraph_to_frame(tf_left, "Resource efficiency.", size=14, bold=True, color=SLATE_900, space_before=Pt(14))
add_paragraph_to_frame(tf_left, "Transfer learning yields +8.58% higher accuracy compared to training from scratch (R-FL) while converging in nearly 4× less compute time.", size=13, color=SLATE_700, space_before=Pt(2))

# Right Column: Future Directions
tb_right = add_textbox(slide, Inches(6.8), Inches(1.6), Inches(5.8), Inches(4.5),
                       "Future directions", size=15, bold=True, color=SLATE_400)
tf_right = tb_right.text_frame

add_numbered_item(tf_right, "1", "Vary frozen boundary", "Test fine-tuning with fewer or more frozen layers (e.g. freeze only layer1-2).")
add_numbered_item(tf_right, "2", "Scale model scale", "Evaluate capacity-matched controls on deeper networks (ResNet-50) or ViT backbones.")
add_numbered_item(tf_right, "3", "Progressive unfreezing", "Gradually unfreeze earlier layers with a smaller learning rate as training progresses.")
add_numbered_item(tf_right, "4", "Data augmentation", "Investigate if heavy augmentations can close the scratch training accuracy gap.")

add_slide_number(slide, 7)

# Save
prs.save(OUT)
print(f"✅ Generated minimal professional presentation matching reference style at: {OUT}")
