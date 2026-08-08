"""
Generate two figures from existing results (no retraining, CPU-only):
  1) results/learning_curves.png  -- val accuracy & loss vs epoch, per condition/seed
  2) results/feature_pca.png      -- PCA of penultimate (512-d) features on the test set,
                                     pretrained-frozen vs random-frozen
Run: python3 -m src.make_extra_figures
"""
import os, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.model import get_model
from src.dataset import get_dataset_splits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

COND = {
    "pretrained-frozen": ("Pretrained-Frozen", "#991b1b"),
    "random-full":       ("Random-Full",       "#475569"),
    "random-frozen":     ("Random-Frozen",     "#94a3b8"),
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#eef2f7",
    "grid.linewidth": 1, "figure.dpi": 150,
})


def learning_curves():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1))
    seen = set()
    for cond, (label, color) in COND.items():
        for hp in sorted(glob.glob(os.path.join(RES, f"history_{cond}_seed*.json"))):
            h = json.load(open(hp))
            ep = np.arange(1, len(h["val_acc"]) + 1)
            lab = label if cond not in seen else None
            seen.add(cond)
            a1.plot(ep, np.array(h["val_acc"]) * 100, color=color, lw=1.8, alpha=0.85, label=lab)
            a2.plot(ep, h["val_loss"], color=color, lw=1.8, alpha=0.85, label=lab)
    a1.set_title("Validation accuracy vs. epoch", fontsize=12, weight="bold", color="#1e293b")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Val accuracy (%)"); a1.set_ylim(55, 96)
    a2.set_title("Validation loss vs. epoch", fontsize=12, weight="bold", color="#1e293b")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Val loss")
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(RES, "learning_curves.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


@torch.no_grad()
def extract_features(condition, seed, loader, device="cpu", max_batches=None):
    model = get_model(condition=condition, num_classes=6, seed=seed)
    ck = torch.load(os.path.join(RES, f"best_model_{condition}_seed{seed}.pth"), map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    model.eval().to(device)
    feats, labels = [], []
    box = {}
    h = model.avgpool.register_forward_hook(lambda m, i, o: box.__setitem__("f", o))
    for bi, (x, y) in enumerate(loader):
        model(x.to(device))
        feats.append(box["f"].squeeze(-1).squeeze(-1).cpu().numpy())
        labels.append(y.numpy())
        if max_batches and bi + 1 >= max_batches:
            break
    h.remove()
    return np.concatenate(feats), np.concatenate(labels)


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S[:2] ** 2) / (S ** 2).sum()


def feature_pca(seed=42):
    _, _, test_loader, classes = get_dataset_splits(DATA, batch_size=64, num_workers=4, seed=42)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [("pretrained-frozen", "Pretrained-Frozen features"),
              ("random-frozen", "Random-Frozen features")]
    for ax, (cond, title) in zip(axes, panels):
        X, y = extract_features(cond, seed, test_loader)
        Z, var = pca2(X)
        for ci, cname in enumerate(classes):
            m = y == ci
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.55, color=cmap(ci), label=cname.capitalize())
        ax.set_title(f"{title}\n(PC1+PC2 = {100*var.sum():.0f}% var)", fontsize=11.5,
                     weight="bold", color="#1e293b")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(True)
    axes[1].legend(frameon=False, fontsize=8.5, markerscale=1.6,
                   loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("PCA of penultimate (512-d) features on the test split  ·  seed 42",
                 fontsize=12.5, weight="bold", color="#1e293b", y=1.02)
    fig.tight_layout()
    out = os.path.join(RES, "feature_pca.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def ablation_curve():
    d = json.load(open(os.path.join(RES, "ablation", "ablation_results.json")))
    order = ["fc", "layer4", "layer3", "layer2", "full"]
    xlab = ["fc only\n(linear probe)", "layer4\n(paper PT-F)", "layer3→fc", "layer2→fc", "full\nfine-tune"]
    by = {r["level"]: r for r in d["results"]}
    acc = [by[l]["test_acc"] * 100 for l in order]
    prm = [by[l]["trainable_params"] for l in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(x, acc, "-", color="#94a3b8", lw=2, zorder=1)
    for i, (a, p) in enumerate(zip(acc, prm)):
        hero = order[i] == "layer4"
        ax.scatter(x[i], a, s=150 if hero else 90, zorder=3,
                   color="#991b1b" if hero else "#475569",
                   edgecolor="white", linewidth=1.5)
        ax.annotate(f"{a:.1f}%", (x[i], a), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10.5, weight="bold",
                    color="#991b1b" if hero else "#1e293b")
        pl = f"{p/1e6:.1f}M" if p >= 1e6 else f"{p:,}"
        ax.annotate(pl, (x[i], a), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8.5, color="#64748b")
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=9.5)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(90.3, 93.6)
    ax.set_title("Freeze-depth ablation: accuracy vs. how much of the network is trainable\n"
                 "(pretrained ResNet-18, 40% train subset, seed 42)",
                 fontsize=11.5, weight="bold", color="#1e293b")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.08)
    fig.tight_layout()
    out = os.path.join(RES, "ablation_curve.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def gabor_curve():
    d = json.load(open(os.path.join(RES, "ablation", "gabor_ablation_results.json")))
    order = ["fc", "l1", "l2", "l3", "l4"]
    xlab = ["fc only\n(read-out)", "+layer1", "+layer2", "+layer3", "+layer4\n(all but conv1)"]
    by = {}
    for r in d["results"]:
        by[(r["front_end"], r["level"])] = r
    x = np.arange(len(order))
    styles = {"learned": ("Learned conv1 (frozen)", "#991b1b", "o"),
              "gabor":   ("Gabor conv1 (frozen)",   "#2563a8", "s")}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for fe, (label, color, marker) in styles.items():
        acc = [by[(fe, l)]["test_acc"] * 100 for l in order]
        ax.plot(x, acc, "-", color=color, lw=2, marker=marker, ms=8,
                markeredgecolor="white", markeredgewidth=1.3, label=label, zorder=3)
        for i, a in enumerate(acc):
            dy = 11 if fe == "learned" else -17
            ax.annotate(f"{a:.1f}", (x[i], a), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=9, weight="bold", color=color)
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=9.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Trainable region (conv1 always frozen; unfrozen from the front)")
    ax.set_title("How far up does the first layer matter?\n"
                 "Freeze-depth sweep with a fixed Gabor vs. learned first conv "
                 "(ResNet-18, full data, seed 42)",
                 fontsize=11.5, weight="bold", color="#1e293b")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.08)
    fig.tight_layout()
    out = os.path.join(RES, "gabor_curve.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    if "--ablation" in sys.argv:
        ablation_curve()
    elif "--gabor" in sys.argv:
        gabor_curve()
    else:
        learning_curves()
        feature_pca()
