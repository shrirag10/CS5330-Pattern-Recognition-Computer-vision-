"""
Demo video for the CS 5330 final project.

Two segments, silent, with burned-in captions so it stands alone muted:
  1. Live inference -- the Pretrained-Frozen and Random-Frozen checkpoints classify
     the same held-out test images side by side. Both train the identical 8.40M
     parameters, so every disagreement is attributable to initialization alone.
  2. Results tour -- Ken Burns pass over the figures already in results/.

Images are drawn from data/seg_test (the labeled held-out split the reported numbers
come from) so ground truth can be shown and predictions marked correct or incorrect.
The sample is a fixed-seed random draw, not hand-picked.

Run: python3 presentation/make_demo_video.py   ->  presentation/demo_video.mp4
Options: --n-images 18  --fps 30  --seed 42  --dry-run
"""
import os
import sys
import argparse
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import get_model
from src.dataset import get_transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_video.mp4")

W, H = 1920, 1080
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LABELS = [c.capitalize() for c in CLASSES]

# palette shared with make_pptx.py
ACCENT = (0x99, 0x1B, 0x1B)
INK = (0x1E, 0x29, 0x3B)
MUTED = (0x64, 0x74, 0x8B)
FAINT = (0x94, 0xA3, 0xB8)
RULE = (0xE5, 0xE7, 0xEB)
PANEL = (0xF8, 0xFA, 0xFC)
SLATE6 = (0x47, 0x55, 0x69)
WHITE = (0xFF, 0xFF, 0xFF)
GOOD = (0x15, 0x80, 0x3D)
BAD = (0xB9, 0x1C, 0x1C)

_FDIR = None
for _cand in [
    "/home/shrirag10/.local/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf",
    "/usr/share/fonts/truetype/dejavu",
]:
    if os.path.isdir(_cand):
        _FDIR = _cand
        break
if _FDIR is None:
    raise SystemExit("Could not locate DejaVu fonts.")

_CACHE = {}


def font(kind, size):
    key = (kind, size)
    if key not in _CACHE:
        name = {"r": "DejaVuSans.ttf", "b": "DejaVuSans-Bold.ttf",
                "s": "DejaVuSerif.ttf", "sb": "DejaVuSerif-Bold.ttf",
                "m": "DejaVuSansMono.ttf"}[kind]
        path = os.path.join(_FDIR, name)
        if not os.path.exists(path):  # serif may be absent in some installs
            path = os.path.join(_FDIR, "DejaVuSans-Bold.ttf" if kind in ("sb", "b")
                                else "DejaVuSans.ttf")
        _CACHE[key] = ImageFont.truetype(path, size)
    return _CACHE[key]


def ease(t):
    """Smoothstep on [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def blend(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def canvas():
    return Image.new("RGB", (W, H), WHITE)


def text(d, xy, s, f, fill, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def caption(d, line1, line2=None, alpha=1.0):
    """Bottom caption bar."""
    if alpha <= 0.01:
        return
    y0 = 968
    d.rectangle([0, y0, W, H], fill=blend(WHITE, PANEL, alpha))
    d.rectangle([0, y0, W, y0 + 2], fill=blend(WHITE, RULE, alpha))
    d.rectangle([120, y0 + 26, 126, y0 + 82], fill=blend(WHITE, ACCENT, alpha))
    text(d, (152, y0 + 24), line1, font("b", 25), blend(WHITE, INK, alpha))
    if line2:
        text(d, (152, y0 + 60), line2, font("r", 22), blend(WHITE, SLATE6, alpha))


def header(d, kicker_txt, title_txt):
    text(d, (120, 74), kicker_txt.upper(), font("b", 20), ACCENT)
    text(d, (120, 112), title_txt, font("sb", 46), INK)
    d.rectangle([120, 186, 320, 190], fill=ACCENT)


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
def load_model(condition, seed, device):
    ck_path = os.path.join(RES, f"best_model_{condition}_seed{seed}.pth")
    if not os.path.exists(ck_path):
        raise SystemExit(f"Missing checkpoint: {ck_path}")
    model = get_model(condition=condition, num_classes=6, seed=seed)
    ck = torch.load(ck_path, map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    return model.eval().to(device)


def gather_predictions(n_images, seed, device):
    """Run both models over a fixed-seed random sample of the labeled test split."""
    tf = get_transforms()
    rng = np.random.RandomState(seed)

    paths, truths = [], []
    for ci, cname in enumerate(CLASSES):
        cdir = os.path.join(DATA, "seg_test", cname)
        files = sorted(os.listdir(cdir))
        for f in rng.choice(files, size=int(np.ceil(n_images / 6)), replace=False):
            paths.append(os.path.join(cdir, f))
            truths.append(ci)
    order = rng.permutation(len(paths))[:n_images]
    paths = [paths[i] for i in order]
    truths = [truths[i] for i in order]

    ptf = load_model("pretrained-frozen", 42, device)
    rf = load_model("random-frozen", 42, device)

    items = []
    with torch.no_grad():
        for p, t in zip(paths, truths):
            raw = Image.open(p).convert("RGB")
            x = tf(raw).unsqueeze(0).to(device)
            probs = {}
            for tag, m in (("ptf", ptf), ("rf", rf)):
                probs[tag] = F.softmax(m(x), dim=1)[0].cpu().numpy()
            items.append({"image": raw.copy(), "truth": t,
                          "ptf": probs["ptf"], "rf": probs["rf"]})
            raw.close()
    return items


def draw_model_panel(d, x, y, w, title, sub, probs, truth, reveal, accent):
    """One model's 6-class distribution. `reveal` in [0,1] animates the bars."""
    d.rectangle([x, y, x + w, y + 336], fill=PANEL)
    d.rectangle([x, y, x + 6, y + 336], fill=accent)

    text(d, (x + 26, y + 18), title, font("b", 26), INK)
    text(d, (x + 26, y + 52), sub, font("r", 19), MUTED)

    pred = int(np.argmax(probs))
    if reveal > 0.985:
        ok = pred == truth
        text(d, (x + w - 26, y + 26), "CORRECT" if ok else "INCORRECT",
             font("b", 24), GOOD if ok else BAD, anchor="ra")

    bx, bw = x + 200, w - 300
    for i in range(6):
        ry = y + 96 + i * 39
        is_pred = i == pred
        is_true = i == truth
        lab_col = INK if is_pred else MUTED
        text(d, (x + 180, ry + 4), LABELS[i], font("b" if is_pred else "r", 20),
             lab_col, anchor="ra")
        if is_true:
            d.rectangle([x + 30, ry + 5, x + 48, ry + 23], fill=(0xCB, 0xD5, 0xE1))
        d.rectangle([bx, ry, bx + bw, ry + 26], fill=(0xEC, 0xEF, 0xF3))
        fill_w = int(bw * float(probs[i]) * reveal)
        if fill_w > 0:
            d.rectangle([bx, ry, bx + fill_w, ry + 26],
                        fill=accent if is_pred else FAINT)
        if reveal > 0.35:
            a = ease((reveal - 0.35) / 0.65)
            text(d, (bx + bw + 18, ry + 3), f"{probs[i] * 100:4.1f}%",
                 font("b" if is_pred else "r", 19),
                 blend(WHITE, INK if is_pred else MUTED, a))


def inference_frame(item, reveal, tally, idx, total, cap):
    img = canvas()
    d = ImageDraw.Draw(img)
    header(d, "Live inference", "Same image, same trainable capacity")

    thumb = item["image"].resize((470, 470), Image.LANCZOS)
    img.paste(thumb, (120, 250))
    d.rectangle([120, 250, 590, 720], outline=RULE, width=2)

    text(d, (120, 738), "Ground truth", font("r", 20), MUTED)
    text(d, (120, 766), LABELS[item["truth"]], font("sb", 34), INK)
    text(d, (590, 744), f"{idx + 1} / {total}", font("b", 22), FAINT, anchor="ra")

    d.rectangle([120, 820, 590, 822], fill=RULE)
    for j, line in enumerate([
        "Held-out test split, never seen in training.",
        "Bars show the full softmax over six classes.",
        "Grey marker on a row = the true class.",
        "Both models: seed 42, identical trainable set.",
    ]):
        text(d, (120, 842 + j * 30), line, font("r", 19), MUTED)

    draw_model_panel(d, 680, 250, 1120, "Pretrained-Frozen",
                     "ImageNet init  ·  8.40M trainable", item["ptf"],
                     item["truth"], reveal, ACCENT)
    draw_model_panel(d, 680, 614, 1120, "Random-Frozen",
                     "Random init  ·  8.40M trainable", item["rf"],
                     item["truth"], reveal, SLATE6)

    if tally[0] + tally[1] > 0:
        n = tally[0] + tally[1]
        text(d, (1800, 200), f"Running: PT-F {tally[0]}/{n}   ·   R-F {tally[1]}/{n}",
             font("b", 22), MUTED, anchor="ra")
    caption(d, *cap)
    return img


# ----------------------------------------------------------------------------
# cards and figure tour
# ----------------------------------------------------------------------------
def title_frame(t):
    img = canvas()
    d = ImageDraw.Draw(img)
    a = ease(t / 0.25) if t < 0.25 else 1.0
    d.rectangle([0, 0, W, 12], fill=blend(WHITE, ACCENT, a))
    text(d, (W // 2, 336), "CS 5330  ·  FINAL PROJECT", font("b", 24),
         blend(WHITE, ACCENT, a), anchor="ma")
    text(d, (W // 2, 400), "Transfer Learning versus Training from Scratch",
         font("sb", 62), blend(WHITE, INK, a), anchor="ma")
    text(d, (W // 2, 484), "for Scene Image Classification", font("sb", 62),
         blend(WHITE, INK, a), anchor="ma")
    a2 = ease((t - 0.3) / 0.3) if t > 0.3 else 0.0
    d.rectangle([W // 2 - 90, 596, W // 2 + 90, 599], fill=blend(WHITE, RULE, a2))
    text(d, (W // 2, 636), "Shriman Raghav Srinivasan", font("r", 30),
         blend(WHITE, SLATE6, a2), anchor="ma")
    text(d, (W // 2, 684), "Khoury College of Computer Sciences, Northeastern University",
         font("r", 23), blend(WHITE, MUTED, a2), anchor="ma")
    a3 = ease((t - 0.55) / 0.3) if t > 0.55 else 0.0
    text(d, (W // 2, 790),
         "ResNet-18  ·  Intel Image Classification  ·  6 classes  ·  3 seeds",
         font("b", 24), blend(WHITE, FAINT, a3), anchor="ma")
    return img


def takeaway_frame(t):
    img = canvas()
    d = ImageDraw.Draw(img)
    header(d, "Conclusion", "What the capacity-matched control proved")
    rows = [
        ("93.27%", "Pretrained-Frozen", "ImageNet init, layer4 + fc trainable", ACCENT),
        ("66.54%", "Random-Frozen", "Random init, identical 8.40M trainable", SLATE6),
        ("84.69%", "Random-Full", "Random init, all 11.18M trainable", FAINT),
    ]
    y = 288
    for i, (num, name, sub, col) in enumerate(rows):
        a = ease((t - 0.12 * i) / 0.3)
        if a <= 0.01:
            y += 168
            continue
        text(d, (120, y), num, font("sb", 72), blend(WHITE, col, a))
        text(d, (470, y + 8), name, font("b", 34), blend(WHITE, INK, a))
        text(d, (470, y + 56), sub, font("r", 24), blend(WHITE, MUTED, a))
        d.rectangle([120, y + 112, 1800, y + 113], fill=blend(WHITE, RULE, a))
        y += 168
    a = ease((t - 0.40) / 0.20)
    if a > 0.01:
        text(d, (120, 810), "+26.7 points from pretrained features alone,",
             font("sb", 40), blend(WHITE, ACCENT, a))
        text(d, (120, 866), "with trainable parameter count held fixed.",
             font("sb", 40), blend(WHITE, INK, a))
    return img


FIGURES = [
    ("learning_curves.png", "Pretraining wins from the first epoch",
     "Not just a higher endpoint.",
     "Pretrained-Frozen peaks at epoch 2-4; training from scratch needs 7-12."),
    ("confusion_matrices.png", "The same confusions, at very different scale",
     "Glacier/mountain and buildings/street overlap in all three conditions.",
     "The errors track genuine visual ambiguity in the data, not a broken training setup."),
    ("feature_pca.png", "Why it works, in feature space",
     "Penultimate 512-d features, projected to two principal components.",
     "Pretrained features fall into class-separable clusters; random frozen features stay mixed."),
    ("ablation_curve.png", "Almost none of the benefit needs fine-tuning",
     "A bare linear probe already reaches 91.3%.",
     "Unfreezing past layer4 buys nothing on this task. 40% train subset, seed 42."),
    ("gabor_curve.png", "How far up does the first layer matter?",
     "A fixed Gabor front end plateaus about 7 points low.",
     "No amount of unfreezing the stack above it closes the gap. Full data, seed 42."),
]


def figure_frames(fig_path, t, headline, cap1, cap2):
    """Ken Burns: gentle zoom with fade in/out, figure fitted into the frame."""
    img = canvas()
    d = ImageDraw.Draw(img)
    src = Image.open(fig_path).convert("RGB")

    box_w, box_h = 1680, 700
    zoom = 1.0 + 0.05 * t
    scale = min(box_w / src.width, box_h / src.height) * zoom
    nw, nh = max(1, int(src.width * scale)), max(1, int(src.height * scale))
    scaled = src.resize((nw, nh), Image.LANCZOS)
    src.close()

    # centre-crop anything that overflows the box
    left = max(0, (nw - box_w) // 2)
    top = max(0, (nh - box_h) // 2)
    scaled = scaled.crop((left, top, min(nw, left + box_w), min(nh, top + box_h)))

    fade = min(ease(t / 0.10), ease((1.0 - t) / 0.10))
    if fade < 0.999:
        scaled = Image.blend(Image.new("RGB", scaled.size, WHITE), scaled, fade)

    header(d, "Results", headline)
    img.paste(scaled, ((W - scaled.width) // 2, 232 + (box_h - scaled.height) // 2))
    caption(d, cap1, cap2)
    return img


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------
def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    for name, *_ in FIGURES:
        p = os.path.join(RES, name)
        if not os.path.exists(p):
            raise SystemExit(f"Missing figure: {p}\nRun src.make_extra_figures first.")

    print(f"running both models over {args.n_images} test images...")
    items = gather_predictions(args.n_images, args.seed, device)
    ptf_ok = sum(int(np.argmax(i["ptf"]) == i["truth"]) for i in items)
    rf_ok = sum(int(np.argmax(i["rf"]) == i["truth"]) for i in items)
    print(f"  sample accuracy: PT-F {ptf_ok}/{len(items)}  R-F {rf_ok}/{len(items)}")

    fps = args.fps
    plan = []  # (kind, payload, n_frames)

    plan.append(("title", None, int(5.5 * fps)))

    caps = [
        ("Both models train the identical 8.40M parameters.",
         "Only the initialization differs -- ImageNet weights versus random."),
        ("Grey tick marks the ground-truth class.",
         "Bars are the full softmax distribution over all six classes."),
        ("Pretrained-Frozen is right far more often.",
         "And when it is unsure, the probability mass still lands on plausible classes."),
        ("This is a fixed-seed random sample, not hand-picked.",
         "Over the full 3,000-image test split: 93.27% versus 66.54%."),
    ]
    per = max(1, len(items) // len(caps))
    tally = [0, 0]
    for i, it in enumerate(items):
        plan.append(("infer", (it, i, len(items), tuple(tally),
                               caps[min(i // per, len(caps) - 1)]), int(2.7 * fps)))
        tally[0] += int(np.argmax(it["ptf"]) == it["truth"])
        tally[1] += int(np.argmax(it["rf"]) == it["truth"])

    for name, headline, cap1, cap2 in FIGURES:
        plan.append(("figure", (os.path.join(RES, name), headline, cap1, cap2),
                     int(9.0 * fps)))

    plan.append(("takeaway", None, int(10.0 * fps)))

    total = sum(n for _, _, n in plan)
    print(f"total {total} frames = {total / fps:.1f}s at {fps}fps")
    if args.dry_run:
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    done = 0
    try:
        for kind, payload, n in plan:
            for k in range(n):
                t = k / max(1, n - 1)
                if kind == "title":
                    frame = title_frame(t)
                elif kind == "takeaway":
                    frame = takeaway_frame(t)
                elif kind == "figure":
                    frame = figure_frames(payload[0], t, payload[1], payload[2], payload[3])
                else:
                    it, idx, tot, tal, cap = payload
                    # 0.45s settle, 1.1s bar fill, rest hold
                    reveal = ease((t * n / fps - 0.45) / 1.1)
                    frame = inference_frame(it, reveal, tal, idx, tot, cap)
                proc.stdin.write(frame.tobytes())
                done += 1
                if done % 150 == 0:
                    print(f"  {done}/{total} frames ({100 * done / total:.0f}%)", flush=True)
        proc.stdin.close()
    except BrokenPipeError:
        raise SystemExit("ffmpeg closed the pipe early -- encoding failed.")
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, {total / fps:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    build(ap.parse_args())
