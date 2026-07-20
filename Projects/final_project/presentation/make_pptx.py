"""
Generate a professional .pptx presentation for the CS 5330 final project.
Clean, minimal design — white slides, simple typography, embedded charts.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os, json
import numpy as np

ASSETS = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation/assets"
RESULTS = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
OUT = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation/presentation.pptx"

# Colors
INDIGO  = RGBColor(0x4F, 0x46, 0xE5)
AMBER   = RGBColor(0xD9, 0x77, 0x06)
GREEN   = RGBColor(0x05, 0x96, 0x69)
DARK    = RGBColor(0x0F, 0x17, 0x2A)
BODY    = RGBColor(0x47, 0x55, 0x69)
MUTED   = RGBColor(0x94, 0xA3, 0xB8)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
BORDER  = RGBColor(0xE2, 0xE8, 0xF0)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


# ─── Helpers ───

def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, left, top, width, height, fill_color, border_color=None, radius=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def set_text(shape, text, size=14, bold=False, color=BODY, align=PP_ALIGN.LEFT, font_name='Calibri'):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return tf

def add_textbox(slide, left, top, width, height, text, size=14, bold=False, color=BODY, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    set_text(txBox, text, size, bold, color, align)
    return txBox

def add_para(tf, text, size=14, bold=False, color=BODY, space_before=Pt(4), space_after=Pt(2)):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = 'Calibri'
    p.space_before = space_before
    p.space_after = space_after
    return p

def add_slide_label(slide, text):
    add_textbox(slide, Inches(0.7), Inches(0.4), Inches(4), Inches(0.3),
                text, size=10, bold=True, color=INDIGO)

def add_title(slide, text):
    add_textbox(slide, Inches(0.7), Inches(0.7), Inches(10), Inches(0.6),
                text, size=28, bold=True, color=DARK)

def add_slide_number(slide, num, total=12):
    add_textbox(slide, Inches(12.0), Inches(7.0), Inches(1), Inches(0.3),
                f"{num} / {total}", size=10, bold=False, color=MUTED, align=PP_ALIGN.RIGHT)

def add_accent_line(slide, left, top, width=Inches(0.6)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(3))
    shape.fill.solid()
    shape.fill.fore_color.rgb = INDIGO
    shape.line.fill.background()


# ═══════════════════════════════════════════
# SLIDE 1 — Title
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide)

# Centered content
add_textbox(slide, Inches(1.5), Inches(1.4), Inches(10.3), Inches(0.3),
            "CS 5330  ·  PATTERN RECOGNITION & COMPUTER VISION", size=11, bold=True, color=INDIGO, align=PP_ALIGN.CENTER)

add_textbox(slide, Inches(1.5), Inches(2.2), Inches(10.3), Inches(1.2),
            "Transfer Learning vs.\nTraining from Scratch", size=42, bold=True, color=DARK, align=PP_ALIGN.CENTER)

# Accent line
add_accent_line(slide, Inches(6.15), Inches(3.6), Inches(1))

add_textbox(slide, Inches(2), Inches(3.9), Inches(9.3), Inches(0.5),
            "A Capacity-Matched Empirical Study Using ResNet-18", size=17, bold=False, color=BODY, align=PP_ALIGN.CENTER)

add_textbox(slide, Inches(2), Inches(5.0), Inches(9.3), Inches(0.8),
            "Shrirag Trolihal Santhosh\nNortheastern University  ·  Khoury College of Computer Sciences  ·  Summer 2026",
            size=14, bold=False, color=MUTED, align=PP_ALIGN.CENTER)

add_slide_number(slide, 1)


# ═══════════════════════════════════════════
# SLIDE 2 — Problem Statement
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "MOTIVATION")
add_title(slide, "Problem Statement")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Left column — text
tb = add_textbox(slide, Inches(0.7), Inches(1.6), Inches(5.8), Inches(0.5),
                 "Does transfer learning outperform training from scratch because of better features, or simply because it has more trainable parameters?",
                 size=15, bold=False, color=DARK)

tb2 = add_textbox(slide, Inches(0.7), Inches(2.5), Inches(5.8), Inches(1.5),
                  "Standard comparisons confound two factors:", size=14, color=BODY)
tf = tb2.text_frame
add_para(tf, "•  Quality of pretrained representations", size=13, color=BODY)
add_para(tf, "•  Reduced trainable capacity from layer freezing", size=13, color=BODY)
add_para(tf, "", size=8)
add_para(tf, "We design a three-condition experiment to isolate each factor independently.", size=14, bold=True, color=DARK)

# Right column — 3 condition cards
card_x = Inches(7.0)
card_w = Inches(5.8)
card_h = Inches(1.2)

# PT-F card
c1 = add_rect(slide, card_x, Inches(1.6), card_w, card_h, LIGHT_BG, BORDER)
bar1 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, card_x, Inches(1.6), Pt(4), card_h)
bar1.fill.solid(); bar1.fill.fore_color.rgb = INDIGO; bar1.line.fill.background()
add_textbox(slide, card_x + Inches(0.2), Inches(1.7), card_w - Inches(0.3), Inches(0.3),
            "Pretrained-Frozen (PT-F)", size=14, bold=True, color=DARK)
add_textbox(slide, card_x + Inches(0.2), Inches(2.05), card_w - Inches(0.3), Inches(0.4),
            "ImageNet weights · Early layers frozen · Only layer4 + fc trained", size=12, color=BODY)

# R-F card
c2 = add_rect(slide, card_x, Inches(3.0), card_w, card_h, LIGHT_BG, BORDER)
bar2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, card_x, Inches(3.0), Pt(4), card_h)
bar2.fill.solid(); bar2.fill.fore_color.rgb = AMBER; bar2.line.fill.background()
add_textbox(slide, card_x + Inches(0.2), Inches(3.1), card_w - Inches(0.3), Inches(0.3),
            "Random-Frozen (R-F)", size=14, bold=True, color=DARK)
add_textbox(slide, card_x + Inches(0.2), Inches(3.45), card_w - Inches(0.3), Inches(0.4),
            "Random weights · Early layers frozen · Same capacity as PT-F", size=12, color=BODY)

# R-FL card
c3 = add_rect(slide, card_x, Inches(4.4), card_w, card_h, LIGHT_BG, BORDER)
bar3 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, card_x, Inches(4.4), Pt(4), card_h)
bar3.fill.solid(); bar3.fill.fore_color.rgb = GREEN; bar3.line.fill.background()
add_textbox(slide, card_x + Inches(0.2), Inches(4.5), card_w - Inches(0.3), Inches(0.3),
            "Random-Full (R-FL)", size=14, bold=True, color=DARK)
add_textbox(slide, card_x + Inches(0.2), Inches(4.85), card_w - Inches(0.3), Inches(0.4),
            "Random weights · All layers trainable · Full-scratch baseline", size=12, color=BODY)

add_slide_number(slide, 2)


# ═══════════════════════════════════════════
# SLIDE 3 — Dataset
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "DATA")
add_title(slide, "Intel Image Classification Dataset")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Stats boxes
for i, (val, lbl) in enumerate([("14,034", "Train Images"), ("3,000", "Test Images"), ("6", "Classes")]):
    x = Inches(0.7 + i * 1.6)
    box = add_rect(slide, x, Inches(1.6), Inches(1.4), Inches(0.9), LIGHT_BG, BORDER)
    add_textbox(slide, x + Inches(0.1), Inches(1.65), Inches(1.2), Inches(0.4),
                val, size=22, bold=True, color=INDIGO, align=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.1), Inches(2.05), Inches(1.2), Inches(0.3),
                lbl, size=9, bold=True, color=MUTED, align=PP_ALIGN.CENTER)

# Text info
tb = add_textbox(slide, Inches(0.7), Inches(2.8), Inches(4.5), Inches(0.3),
                 "Classes", size=15, bold=True, color=DARK)
add_textbox(slide, Inches(0.7), Inches(3.1), Inches(4.5), Inches(0.3),
            "Buildings · Forest · Glacier · Mountain · Sea · Street", size=13, color=BODY)

tb2 = add_textbox(slide, Inches(0.7), Inches(3.6), Inches(4.5), Inches(0.3),
                  "Preprocessing", size=15, bold=True, color=DARK)
tb3 = add_textbox(slide, Inches(0.7), Inches(3.9), Inches(4.5), Inches(1.5), "", size=13, color=BODY)
tf = tb3.text_frame
tf.paragraphs[0].text = "•  Resize shorter side → 224 px"
tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.color.rgb = BODY; tf.paragraphs[0].font.name = 'Calibri'
add_para(tf, "•  Center crop to 224 × 224", size=13, color=BODY)
add_para(tf, "•  ImageNet mean/std normalization", size=13, color=BODY)
add_para(tf, "•  Stratified 90/10 train-val split", size=13, color=BODY)

# Sample images
slide.shapes.add_picture(os.path.join(ASSETS, "samples.png"),
                         Inches(5.6), Inches(1.6), Inches(7.2))

add_slide_number(slide, 3)


# ═══════════════════════════════════════════
# SLIDE 4 — Architecture
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "ARCHITECTURE")
add_title(slide, "ResNet-18 Layer Partitioning")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Architecture blocks
block_data = [
    ("conv1\nbn1", INDIGO, 0.7, 1.4),
    ("layer1", INDIGO, 2.4, 1.0),
    ("layer2", INDIGO, 3.7, 1.0),
    ("layer3", INDIGO, 5.0, 1.0),
    ("layer4", GREEN, 6.6, 1.2),
    ("avgpool", GREEN, 8.1, 0.9),
    ("fc (6)", GREEN, 9.3, 1.0),
]
block_y = Inches(1.7)
block_h = Inches(0.9)
for (label, color, x_in, w_in) in block_data:
    shape = add_rect(slide, Inches(x_in), block_y, Inches(w_in), block_h, color)
    set_text(shape, label, size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

# Arrows between blocks
for x_in in [2.1, 3.4, 4.7, 6.0, 7.8, 9.0]:
    add_textbox(slide, Inches(x_in), block_y + Inches(0.2), Inches(0.3), Inches(0.4),
                "→", size=16, color=MUTED, align=PP_ALIGN.CENTER)

# Legend
add_textbox(slide, Inches(0.7), Inches(2.8), Inches(4), Inches(0.3),
            "■ Frozen θf — 2,782,784 params          ■ Trainable θt — 8,396,806 params",
            size=11, color=BODY)

# BN callout
callout = add_rect(slide, Inches(0.7), Inches(3.3), Inches(11.9), Inches(0.7), RGBColor(0xF0, 0xF4, 0xFF))
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(3.3), Pt(3), Inches(0.7))
bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
add_textbox(slide, Inches(1.0), Inches(3.4), Inches(11.4), Inches(0.5),
            "BN Handling: All Batch Normalization layers in θf are locked in eval() mode during training to prevent running statistics from drifting.",
            size=12, bold=False, color=DARK)

# Comparison table
table_data = [
    ["Comparison", "Controls for", "Isolates"],
    ["PT-F  vs  R-F", "Same trainable capacity", "Feature quality"],
    ["R-F  vs  R-FL", "Same initialization", "Model capacity"],
    ["PT-F  vs  R-FL", "—", "Overall: transfer vs scratch"],
]
tbl = slide.shapes.add_table(len(table_data), 3, Inches(0.7), Inches(4.4), Inches(7), Inches(1.6)).table
tbl.columns[0].width = Inches(2.2)
tbl.columns[1].width = Inches(2.4)
tbl.columns[2].width = Inches(2.4)
for r, row in enumerate(table_data):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.name = 'Calibri'
            p.font.color.rgb = DARK if r == 0 else BODY
            p.font.bold = (r == 0)
            
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_BG if r == 0 else WHITE

add_slide_number(slide, 4)


# ═══════════════════════════════════════════
# SLIDE 5 — Experimental Setup
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "EXPERIMENTS")
add_title(slide, "Experimental Setup")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Left — config table
cfg = [
    ["Parameter", "Value"],
    ["Optimizer", "SGD"],
    ["Learning Rate", "1 × 10⁻³"],
    ["Momentum", "0.9"],
    ["Weight Decay", "1 × 10⁻⁴"],
    ["Batch Size", "32"],
    ["Early Stopping", "Patience = 5 (val loss)"],
    ["Seeds", "42, 100, 2026"],
    ["Hardware", "NVIDIA RTX 4060"],
]
tbl = slide.shapes.add_table(len(cfg), 2, Inches(0.7), Inches(1.6), Inches(4.8), Inches(3.6)).table
tbl.columns[0].width = Inches(2.0)
tbl.columns[1].width = Inches(2.8)
for r, row in enumerate(cfg):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.name = 'Calibri'
            p.font.bold = (r == 0 or c == 0)
            p.font.color.rgb = DARK if (r == 0 or c == 0) else BODY
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_BG if r == 0 else WHITE

# Right — pipeline flowchart
flow_steps = [
    ("Intel Dataset (14,034 images)", RGBColor(0xE0, 0xE7, 0xFF)),
    ("Resize → Center Crop → Normalize", RGBColor(0xE0, 0xE7, 0xFF)),
    ("Stratified 90/10 Split", RGBColor(0xE0, 0xE7, 0xFF)),
]
add_textbox(slide, Inches(6.2), Inches(1.5), Inches(4), Inches(0.3),
            "Pipeline Overview", size=15, bold=True, color=DARK)

flow_x = Inches(6.2)
flow_w = Inches(6.5)
y_start = 1.9
for i, (label, bg_c) in enumerate(flow_steps):
    y = Inches(y_start + i * 0.7)
    box = add_rect(slide, flow_x, y, flow_w, Inches(0.45), bg_c)
    set_text(box, label, size=12, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    if i < len(flow_steps) - 1:
        add_textbox(slide, Inches(9.2), y + Inches(0.4), Inches(0.5), Inches(0.3),
                    "↓", size=14, color=MUTED, align=PP_ALIGN.CENTER)

# 3 condition boxes
cond_w = Inches(2.0)
cond_y = Inches(y_start + 2.3)
for i, (label, color) in enumerate([("PT-F", INDIGO), ("R-F", AMBER), ("R-FL", GREEN)]):
    x = Inches(6.2 + i * 2.17)
    box = add_rect(slide, x, cond_y, cond_w, Inches(0.4), color)
    set_text(box, label, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_textbox(slide, Inches(6.2), cond_y + Inches(0.5), flow_w, Inches(0.3),
            "↓  × 3 seeds each", size=11, color=MUTED, align=PP_ALIGN.CENTER)

more_steps = [
    "SGD + Early Stopping → Best Checkpoint",
    "Test Evaluation (3,000 images)",
]
for i, label in enumerate(more_steps):
    y = cond_y + Inches(0.8 + i * 0.7)
    box = add_rect(slide, flow_x, y, flow_w, Inches(0.45), RGBColor(0xE0, 0xE7, 0xFF))
    set_text(box, label, size=12, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    if i < len(more_steps) - 1:
        add_textbox(slide, Inches(9.2), y + Inches(0.4), Inches(0.5), Inches(0.3),
                    "↓", size=14, color=MUTED, align=PP_ALIGN.CENTER)

add_slide_number(slide, 5)


# ═══════════════════════════════════════════
# SLIDE 6 — Test Accuracy Results
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "RESULTS")
add_title(slide, "Test Accuracy")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Big stat numbers
stats = [("93.3%", "±0.23%", "Pretrained-Frozen", INDIGO),
         ("84.7%", "±1.87%", "Random-Full", GREEN),
         ("66.5%", "±4.04%", "Random-Frozen", AMBER)]
for i, (val, std, label, color) in enumerate(stats):
    y = Inches(1.7 + i * 1.5)
    box = add_rect(slide, Inches(0.7), y, Inches(5.5), Inches(1.2), LIGHT_BG, BORDER)
    add_textbox(slide, Inches(0.9), y + Inches(0.1), Inches(2.5), Inches(0.6),
                val, size=30, bold=True, color=color)
    add_textbox(slide, Inches(0.9), y + Inches(0.7), Inches(2.5), Inches(0.3),
                label, size=11, bold=True, color=MUTED)
    add_textbox(slide, Inches(3.5), y + Inches(0.7), Inches(1.5), Inches(0.3),
                std, size=11, color=MUTED)
    # Progress bar
    bar_bg = add_rect(slide, Inches(4.0), y + Inches(0.35), Inches(2.0), Inches(0.3), BORDER)
    pct = float(val.replace('%','')) / 100
    bar_fill = add_rect(slide, Inches(4.0), y + Inches(0.35), Inches(2.0 * pct), Inches(0.3), color)

# Bar chart on right
slide.shapes.add_picture(os.path.join(ASSETS, "bar.png"),
                         Inches(6.8), Inches(1.6), Inches(6.0))

add_slide_number(slide, 6)


# ═══════════════════════════════════════════
# SLIDE 7 — Training Curves
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "RESULTS")
add_title(slide, "Training Dynamics")
add_accent_line(slide, Inches(0.7), Inches(1.3))

slide.shapes.add_picture(os.path.join(ASSETS, "curves.png"),
                         Inches(0.7), Inches(1.6), Inches(11.9))

# Callouts at bottom
callout_data = [
    ("PT-F:", "Starts high, converges in ~3 epochs.", INDIGO),
    ("R-F:", "Stagnates — random frozen features bottleneck.", AMBER),
    ("R-FL:", "Steady climb over 12+ epochs. Needs 2× time.", GREEN),
]
for i, (title, desc, color) in enumerate(callout_data):
    x = Inches(0.7 + i * 4.1)
    box = add_rect(slide, x, Inches(5.6), Inches(3.8), Inches(0.8), RGBColor(0xF0, 0xF4, 0xFF))
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(5.6), Pt(3), Inches(0.8))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    tb = add_textbox(slide, x + Inches(0.15), Inches(5.65), Inches(3.5), Inches(0.6),
                     f"{title} {desc}", size=11, color=DARK)

add_slide_number(slide, 7)


# ═══════════════════════════════════════════
# SLIDE 8 — Convergence
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "RESULTS")
add_title(slide, "Convergence Speed")
add_accent_line(slide, Inches(0.7), Inches(1.3))

slide.shapes.add_picture(os.path.join(ASSETS, "convergence.png"),
                         Inches(0.5), Inches(1.6), Inches(7.5))

# Table
conv_data = [
    ["Condition", "Epochs", "Time", "Best Val Acc"],
    ["PT-F", "8.0", "1.7 min", "93.8%"],
    ["R-F", "8.7", "1.9 min", "67.8%"],
    ["R-FL", "15.0", "7.1 min", "86.3%"],
]
tbl = slide.shapes.add_table(4, 4, Inches(8.3), Inches(1.8), Inches(4.5), Inches(1.8)).table
for c_i in range(4): tbl.columns[c_i].width = Inches(1.125)
for r, row in enumerate(conv_data):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12); p.font.name = 'Calibri'
            p.font.bold = (r == 0 or (c == 0 and r > 0))
            p.font.color.rgb = DARK if r == 0 else BODY
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT_BG if r == 0 else WHITE

# Callout
callout = add_rect(slide, Inches(8.3), Inches(4.0), Inches(4.5), Inches(0.7), RGBColor(0xF0, 0xF4, 0xFF))
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(8.3), Inches(4.0), Pt(3), Inches(0.7))
bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
add_textbox(slide, Inches(8.5), Inches(4.1), Inches(4.2), Inches(0.5),
            "Transfer learning achieves higher accuracy in 4× less time than training from scratch.",
            size=12, bold=True, color=DARK)

add_slide_number(slide, 8)


# ═══════════════════════════════════════════
# SLIDE 9 — Confusion Matrices
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "RESULTS")
add_title(slide, "Confusion Matrices")
add_accent_line(slide, Inches(0.7), Inches(1.3))

slide.shapes.add_picture(os.path.join(ASSETS, "confusion.png"),
                         Inches(0.3), Inches(1.5), Inches(12.7))

# Bottom callouts
cb1 = add_rect(slide, Inches(0.7), Inches(5.8), Inches(5.8), Inches(0.7), RGBColor(0xF0, 0xF4, 0xFF))
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(5.8), Pt(3), Inches(0.7))
bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
add_textbox(slide, Inches(0.9), Inches(5.85), Inches(5.4), Inches(0.6),
            "Primary confusions: Glacier ↔ Mountain and Buildings ↔ Street across all conditions.",
            size=11, color=DARK)

cb2 = add_rect(slide, Inches(6.8), Inches(5.8), Inches(5.9), Inches(0.7), RGBColor(0xF0, 0xF4, 0xFF))
bar2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(5.8), Pt(3), Inches(0.7))
bar2.fill.solid(); bar2.fill.fore_color.rgb = GREEN; bar2.line.fill.background()
add_textbox(slide, Inches(7.0), Inches(5.85), Inches(5.5), Inches(0.6),
            "PT-F dramatically reduces off-diagonal errors even for visually similar classes.",
            size=11, color=DARK)

add_slide_number(slide, 9)


# ═══════════════════════════════════════════
# SLIDE 10 — Per-class F1
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "RESULTS")
add_title(slide, "Per-Class F1 Scores")
add_accent_line(slide, Inches(0.7), Inches(1.3))

slide.shapes.add_picture(os.path.join(ASSETS, "f1.png"),
                         Inches(0.5), Inches(1.6), Inches(8.0))

add_textbox(slide, Inches(8.8), Inches(1.8), Inches(4), Inches(0.3),
            "Observations", size=15, bold=True, color=DARK)
obs = add_textbox(slide, Inches(8.8), Inches(2.3), Inches(4.2), Inches(3), "", size=13, color=BODY)
tf = obs.text_frame
tf.paragraphs[0].text = "•  Forest has the highest F1 across all conditions — strong texture cues"
tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.color.rgb = BODY; tf.paragraphs[0].font.name = 'Calibri'
add_para(tf, "•  Glacier and Mountain show lowest F1 — visually similar terrain", size=13, color=BODY)
add_para(tf, "•  PT-F achieves >90% F1 on 5/6 classes", size=13, color=BODY)
add_para(tf, "•  R-F struggles most on Sea and Street", size=13, color=BODY)

add_slide_number(slide, 10)


# ═══════════════════════════════════════════
# SLIDE 11 — Key Findings
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "ANALYSIS")
add_title(slide, "Key Findings")
add_accent_line(slide, Inches(0.7), Inches(1.3))

findings = [
    ("1. Feature Quality Dominates", "PT-F vs R-F (same capacity)", "+26.7%",
     "Pretrained features provide rich, transferable representations. Random frozen features are noise-like.", INDIGO),
    ("2. Capacity Helps, But Less", "R-F vs R-FL (same init)", "+18.2%",
     "Unlocking all parameters helps, but needs 2× longer and falls 8.6% short of PT-F.", GREEN),
    ("3. Transfer Wins Overall", "PT-F vs R-FL", "+8.6% · 4× faster",
     "Pretrained features > more parameters, especially under limited data and compute.", RGBColor(0x7C, 0x3A, 0xED)),
]

for i, (title, subtitle, delta, desc, color) in enumerate(findings):
    x = Inches(0.7 + i * 4.1)
    w = Inches(3.8)
    box = add_rect(slide, x, Inches(1.6), w, Inches(3.4), LIGHT_BG, BORDER)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.6), Pt(4), Inches(3.4))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    
    add_textbox(slide, x + Inches(0.2), Inches(1.7), w - Inches(0.3), Inches(0.3),
                title, size=14, bold=True, color=DARK)
    add_textbox(slide, x + Inches(0.2), Inches(2.05), w - Inches(0.3), Inches(0.25),
                subtitle, size=11, color=MUTED)
    add_textbox(slide, x + Inches(0.2), Inches(2.45), w - Inches(0.3), Inches(0.5),
                delta, size=26, bold=True, color=color)
    add_textbox(slide, x + Inches(0.2), Inches(3.1), w - Inches(0.3), Inches(0.8),
                desc, size=12, color=BODY)

# Bottom callout
callout = add_rect(slide, Inches(0.7), Inches(5.3), Inches(11.9), Inches(0.7), RGBColor(0xF0, 0xF4, 0xFF))
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(5.3), Pt(3), Inches(0.7))
bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
add_textbox(slide, Inches(0.95), Inches(5.35), Inches(11.5), Inches(0.6),
            "Conclusion: The performance gap is driven primarily by feature quality, not model capacity. Our capacity-matched control (R-F) proves this rigorously.",
            size=13, bold=True, color=DARK)

add_slide_number(slide, 11)


# ═══════════════════════════════════════════
# SLIDE 12 — Conclusion
# ═══════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_slide_label(slide, "CONCLUSION")
add_title(slide, "Conclusion & Future Work")
add_accent_line(slide, Inches(0.7), Inches(1.3))

# Summary callout
callout = add_rect(slide, Inches(0.7), Inches(1.6), Inches(11.9), Inches(0.8), RGBColor(0xF0, 0xF4, 0xFF))
bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.6), Pt(3), Inches(0.8))
bar.fill.solid(); bar.fill.fore_color.rgb = INDIGO; bar.line.fill.background()
add_textbox(slide, Inches(0.95), Inches(1.65), Inches(11.4), Inches(0.7),
            "Pretrained feature quality — not model capacity — is the primary driver of transfer learning's superiority in scene classification under limited data.",
            size=14, bold=False, color=DARK)

add_textbox(slide, Inches(0.7), Inches(2.8), Inches(6), Inches(0.3),
            "Future Directions", size=17, bold=True, color=DARK)

futures = [
    ("Vary the partition boundary", "Freeze only layer1-2 instead of layer1-3"),
    ("Extend to larger backbones", "ResNet-50, Vision Transformer (ViT)"),
    ("Progressive unfreezing", "Gradually unlock layers with learning rate warmup"),
    ("Data augmentation", "Apply augmentations to narrow the scratch training gap"),
]
for i, (title, desc) in enumerate(futures):
    x = Inches(0.7 + (i % 2) * 6.2)
    y = Inches(3.3 + (i // 2) * 1.3)
    box = add_rect(slide, x, y, Inches(5.8), Inches(1.0), LIGHT_BG, BORDER)
    add_textbox(slide, x + Inches(0.2), y + Inches(0.1), Inches(5.4), Inches(0.3),
                title, size=13, bold=True, color=DARK)
    add_textbox(slide, x + Inches(0.2), y + Inches(0.45), Inches(5.4), Inches(0.4),
                desc, size=12, color=BODY)

# Thank you
add_textbox(slide, Inches(0.7), Inches(6.2), Inches(12), Inches(0.3),
            "Thank you", size=14, bold=False, color=MUTED, align=PP_ALIGN.CENTER)
add_textbox(slide, Inches(0.7), Inches(6.5), Inches(12), Inches(0.3),
            "Shrirag Trolihal Santhosh  ·  Northeastern University", size=12,
            bold=False, color=MUTED, align=PP_ALIGN.CENTER)

add_slide_number(slide, 12)


# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
prs.save(OUT)
print(f"\n✅ Presentation saved to: {OUT}")
print(f"   {len(prs.slides)} slides, widescreen 16:9")
