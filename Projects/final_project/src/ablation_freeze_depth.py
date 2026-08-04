"""
Freeze-depth ablation (pretrained ResNet-18): how does test accuracy change as we
unfreeze progressively more of the network? Turns the binary frozen/full comparison
into a curve over the freeze boundary.

Freeze levels (trainable set grows left -> right):
  fc      : train only fc                      (linear probe)
  layer4  : train layer4 + fc                  (= main-paper Pretrained-Frozen)
  layer3  : train layer3..fc
  layer2  : train layer2..fc
  full    : train everything                   (full fine-tune)

BN is kept in eval() for stages that are frozen, and left in train() for trainable
stages -- consistent with the paper's BN policy, generalised to each freeze depth.

Compute note: to stay tractable on CPU, the training set is stratified-subsampled to
--frac (default 0.4); the standard validation and test splits are used in full so that
early stopping and the reported test accuracy remain on the full splits. Single seed by
default. On a working GPU, run with --frac 1.0 --seeds 42 100 2026 for full rigor.

Run: python3 -m src.ablation_freeze_depth
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from src.dataset import get_transforms
from src.model import set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "ablation")

# ordered stages, outermost (input) -> innermost (head)
STAGE_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
# freeze level -> first trainable stage (everything from here on is trainable)
LEVELS = {
    "fc":     ["fc"],
    "layer4": ["layer4", "fc"],
    "layer3": ["layer3", "layer4", "fc"],
    "layer2": ["layer2", "layer3", "layer4", "fc"],
    "full":   STAGE_ORDER,
}


def build_model(trainable_stages, seed):
    set_seed(seed)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    set_seed(seed)
    model.fc = nn.Linear(model.fc.in_features, 6)
    tset = set(trainable_stages)
    for name, p in model.named_parameters():
        stage = name.split(".")[0]
        p.requires_grad = stage in tset
    return model, tset


def enforce_bn_eval(model, trainable_stages):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            stage = name.split(".")[0]
            if stage not in trainable_stages:
                mod.eval()


def stratified_train_subset(full_ds, frac, seed):
    targets = np.array(full_ds.targets)
    # reproduce the paper's 10% val split (seed 42), then subsample the remaining train
    rng = np.random.RandomState(42)
    train_idx = []
    for c in range(len(full_ds.classes)):
        ci = np.where(targets == c)[0].copy()
        rng.shuffle(ci)
        train_idx.extend(ci[int(0.10 * len(ci)):])
    train_idx = np.array(sorted(train_idx))
    if frac >= 1.0:
        return train_idx
    rng2 = np.random.RandomState(seed)
    keep = []
    t2 = targets[train_idx]
    for c in range(len(full_ds.classes)):
        ci = train_idx[t2 == c]
        rng2.shuffle(ci)
        keep.extend(ci[:max(1, int(frac * len(ci)))])
    return np.array(sorted(keep))


def val_indices(full_ds):
    targets = np.array(full_ds.targets)
    rng = np.random.RandomState(42)
    vi = []
    for c in range(len(full_ds.classes)):
        ci = np.where(targets == c)[0].copy()
        rng.shuffle(ci)
        vi.extend(ci[:int(0.10 * len(ci))])
    return np.array(sorted(vi))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one(level, seed, loaders, args, device):
    train_loader, val_loader, test_loader = loaders
    trainable = LEVELS[level]
    model, tset = build_model(trainable, seed)
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hist = {"val_acc": [], "val_loss": []}
    best_vl, no_imp, best_state = float("inf"), 0, None
    epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        epochs = epoch
        model.train(); enforce_bn_eval(model, tset)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        # val
        model.eval(); vl = vc = vn = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); vl += crit(out, y).item() * y.size(0)
                vc += (out.argmax(1) == y).sum().item(); vn += y.size(0)
        vl /= vn; va = vc / vn
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"  [{level} seed{seed}] epoch {epoch:02d} val_loss {vl:.4f} val_acc {va*100:.2f}%", flush=True)
        if vl < best_vl:
            best_vl, no_imp = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    return {"level": level, "seed": seed, "trainable_params": int(n_train),
            "epochs_trained": epochs, "best_val_loss": float(best_vl),
            "best_val_acc": float(max(hist["val_acc"])), "test_acc": float(test_acc),
            "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = get_transforms()
    full_train = datasets.ImageFolder(os.path.join(DATA, "seg_train"), transform=tf)
    test_ds = datasets.ImageFolder(os.path.join(DATA, "seg_test"), transform=tf)
    vi = val_indices(full_train)
    val_loader = DataLoader(Subset(full_train, vi), batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    print(f"device={device} frac={args.frac} seeds={args.seeds}", flush=True)
    results = []
    t0 = time.time()
    for seed in args.seeds:
        ti = stratified_train_subset(full_train, args.frac, seed)
        train_loader = DataLoader(Subset(full_train, ti), batch_size=args.batch_size,
                                  shuffle=True, num_workers=4)
        print(f"seed {seed}: {len(ti)} train imgs", flush=True)
        for level in LEVELS:
            rt = time.time()
            r = train_one(level, seed, (train_loader, val_loader, test_loader), args, device)
            r["duration_s"] = round(time.time() - rt, 1)
            results.append(r)
            print(f"DONE {level} seed{seed}: test {r['test_acc']*100:.2f}% "
                  f"({r['trainable_params']:,} params, {r['epochs_trained']} ep, {r['duration_s']}s)", flush=True)
            json.dump({"config": vars(args), "results": results},
                      open(os.path.join(OUT, "ablation_results.json"), "w"), indent=2)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
