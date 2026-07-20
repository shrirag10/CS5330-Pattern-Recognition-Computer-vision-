"""
Generate presentation assets with clean white/light backgrounds
for a professional academic slide deck.
"""
import os, json, random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/data"
RESULTS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
ASSETS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
CLASS_LABELS = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
PTF_COLOR = '#4F46E5'
RF_COLOR  = '#D97706'
RFL_COLOR = '#059669'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ── 1. Sample images ──
def sample_images():
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    fig.patch.set_facecolor('white')
    for col, cls in enumerate(CLASSES):
        d = os.path.join(DATA_DIR, 'seg_train', cls)
        imgs = sorted([f for f in os.listdir(d) if f.endswith(('.jpg','.png'))])
        random.seed(42)
        picks = random.sample(imgs, 2)
        for row, name in enumerate(picks):
            ax = axes[row][col]
            ax.imshow(Image.open(os.path.join(d, name)).convert('RGB'))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor('#e5e7eb'); s.set_linewidth(1)
            if row == 0:
                ax.set_title(CLASS_LABELS[col], fontsize=13, fontweight='600', pad=8)
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(ASSETS_DIR, 'samples.png'), dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ samples.png")

# ── 2. Training curves ──
def training_curves():
    seeds = [42, 100, 2026]
    conds = ['pretrained-frozen', 'random-frozen', 'random-full']
    labels = ['Pretrained-Frozen', 'Random-Frozen', 'Random-Full']
    colors = [PTF_COLOR, RF_COLOR, RFL_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for metric, ax, ylabel, title in [
        ('loss', ax1, 'Validation Loss', 'Validation Loss per Epoch'),
        ('acc',  ax2, 'Validation Accuracy (%)', 'Validation Accuracy per Epoch')
    ]:
        ax.set_facecolor('#fafafa')
        for ci, cond in enumerate(conds):
            all_v = []
            for s in seeds:
                with open(os.path.join(RESULTS_DIR, f'history_{cond}_seed{s}.json')) as f:
                    h = json.load(f)
                v = h[f'val_{metric}']
                if metric == 'acc': v = [x*100 for x in v]
                all_v.append(v)
                ax.plot(range(1, len(v)+1), v, color=colors[ci], alpha=0.2, lw=1)
            ml = min(len(x) for x in all_v)
            avg = [np.mean([x[i] for x in all_v]) for i in range(ml)]
            ax.plot(range(1, ml+1), avg, color=colors[ci], lw=2.5, marker='o',
                    markersize=4, label=labels[ci])
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='600', pad=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb', fontsize=10)
        ax.grid(True, alpha=0.3, color='#d1d5db')
        for s in ax.spines.values(): s.set_edgecolor('#d1d5db')

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(ASSETS_DIR, 'curves.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ curves.png")

# ── 3. Bar chart ──
def bar_chart():
    with open(os.path.join(RESULTS_DIR, 'test_evaluation_metrics.json')) as f:
        d = json.load(f)
    conds = ['pretrained-frozen', 'random-frozen', 'random-full']
    labels = ['Pretrained-\nFrozen', 'Random-\nFrozen', 'Random-\nFull']
    colors = [PTF_COLOR, RF_COLOR, RFL_COLOR]
    means = [d[c]['summary']['mean_accuracy']*100 for c in conds]
    stds  = [d[c]['summary']['std_accuracy']*100 for c in conds]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#fafafa')
    bars = ax.bar(labels, means, yerr=stds, capsize=7, color=colors,
                  edgecolor='white', lw=1.5, width=0.5,
                  error_kw={'elinewidth':2, 'ecolor':'#6b7280'})
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+s+1.2,
                f'{m:.1f}%', ha='center', fontsize=13, fontweight='700')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy by Condition', fontweight='600', pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, color='#d1d5db')
    for s in ax.spines.values(): s.set_edgecolor('#d1d5db')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'bar.png'), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ bar.png")

# ── 4. Confusion matrices ──
def confusion_mats():
    with open(os.path.join(RESULTS_DIR, 'test_evaluation_metrics.json')) as f:
        d = json.load(f)
    conds = ['pretrained-frozen', 'random-frozen', 'random-full']
    titles = ['Pretrained-Frozen', 'Random-Frozen', 'Random-Full']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    short = ['BLD','FOR','GLC','MTN','SEA','STR']

    for i, (c, t) in enumerate(zip(conds, titles)):
        cs = str(d[c]['summary']['closest_seed'])
        cm = np.array(d[c][cs]['confusion_matrix'])
        acc = d[c][cs]['accuracy']*100
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax = axes[i]; ax.set_facecolor('white')
        im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for r in range(6):
            for col in range(6):
                color = 'white' if cmn[r,col] > 0.5 else '#111827'
                ax.text(col, r, str(cm[r][col]), ha='center', va='center',
                        fontsize=9, color=color, fontweight='600')
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(short, fontsize=9)
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel('Predicted')
        if i == 0: ax.set_ylabel('Actual')
        ax.set_title(f'{t}  ({acc:.1f}%)', fontweight='600', pad=10)
        for s in ax.spines.values(): s.set_edgecolor('#d1d5db')

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(ASSETS_DIR, 'confusion.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ confusion.png")

# ── 5. Convergence ──
def convergence():
    with open(os.path.join(RESULTS_DIR, 'summary_metrics.json')) as f:
        d = json.load(f)
    conds = ['pretrained-frozen', 'random-frozen', 'random-full']
    labels = ['PT-Frozen', 'R-Frozen', 'R-Full']
    colors = [PTF_COLOR, RF_COLOR, RFL_COLOR]
    seeds = ['42','100','2026']

    ep = [np.mean([d[c][s]['epochs_trained'] for s in seeds]) for c in conds]
    dur = [np.mean([d[c][s]['duration_seconds'] for s in seeds])/60 for c in conds]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    for ax, vals, yl, tt in [(a1,ep,'Epochs','Epochs to Convergence'),(a2,dur,'Minutes','Training Time')]:
        ax.set_facecolor('#fafafa')
        bs = ax.bar(labels, vals, color=colors, edgecolor='white', lw=1.5, width=0.45)
        for b,v in zip(bs,vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                    f'{v:.1f}', ha='center', fontsize=12, fontweight='600')
        ax.set_ylabel(yl); ax.set_title(tt, fontweight='600', pad=10)
        ax.grid(axis='y', alpha=0.3, color='#d1d5db')
        for s in ax.spines.values(): s.set_edgecolor('#d1d5db')
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(ASSETS_DIR, 'convergence.png'), dpi=180, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ convergence.png")

# ── 6. F1 grouped bar ──
def f1_chart():
    with open(os.path.join(RESULTS_DIR, 'test_evaluation_metrics.json')) as f:
        d = json.load(f)
    conds = ['pretrained-frozen','random-frozen','random-full']
    cl = ['PT-F','R-F','R-FL']
    colors = [PTF_COLOR, RF_COLOR, RFL_COLOR]
    x = np.arange(6); w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#fafafa')
    for i,(c,l,co) in enumerate(zip(conds,cl,colors)):
        cs = str(d[c]['summary']['closest_seed'])
        f1 = [v*100 for v in d[c][cs]['f1']]
        ax.bar(x+i*w, f1, w, label=l, color=co, edgecolor='white', lw=1)
    ax.set_xticks(x+w); ax.set_xticklabels(CLASS_LABELS, fontsize=11)
    ax.set_ylabel('F1 Score (%)'); ax.set_ylim(0, 105)
    ax.set_title('Per-Class F1 Scores', fontweight='600', pad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
    ax.grid(axis='y', alpha=0.3, color='#d1d5db')
    for s in ax.spines.values(): s.set_edgecolor('#d1d5db')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'f1.png'), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ f1.png")

if __name__ == '__main__':
    sample_images()
    training_curves()
    bar_chart()
    confusion_mats()
    convergence()
    f1_chart()
    print("\n✅ All assets regenerated with light theme")
