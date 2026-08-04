"""
Gabor first-layer experiment (professor's suggestion).

Replace ResNet-18's first conv (conv1) with a FIXED Gabor filter bank and freeze it,
then sweep how far up the network we let weights adapt. Compare against the same sweep
with the ordinary learned (ImageNet-pretrained) first layer. The question: how far up
the network does the first-layer choice still matter -- i.e. at what depth does
unfreezing the next stage stop improving test accuracy?

Front ends:
  learned : conv1 = ImageNet-pretrained weights, frozen
  gabor   : conv1 = fixed Gabor bank (8 orientations x 4 wavelengths x 2 phases), frozen

Freeze sweep (conv1 always frozen; bn1 + fc always trainable so the fixed front end can
be re-normalised and read out; residual stages unfrozen as a growing prefix):
  fc  -> train {bn1, fc}
  l1  -> + layer1
  l2  -> + layer1..2
  l3  -> + layer1..3
  l4  -> + layer1..4   (everything except the frozen first conv)

Everything else (rest of the backbone) starts from ImageNet-pretrained weights, so the
experiment measures how much of the pretrained stack must re-adapt when the first layer
is swapped for a fixed front end. BN in frozen residual stages is kept in eval().

Run (GPU): python3 -m src.gabor_ablation --seeds 42
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn
import torchvision.models as models
from src.dataset import get_dataset_splits
from src.model import set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "ablation")

LEVELS = {
    "fc": [],
    "l1": ["layer1"],
    "l2": ["layer1", "layer2"],
    "l3": ["layer1", "layer2", "layer3"],
    "l4": ["layer1", "layer2", "layer3", "layer4"],
}


def gabor_kernel(k, theta, lam, psi, sigma, gamma):
    half = k // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(xr ** 2 + (gamma ** 2) * yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam + psi)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def make_gabor_bank(out_ch=64, k=7, in_ch=3):
    thetas = np.linspace(0, np.pi, 8, endpoint=False)
    lams = [2.5, 3.5, 5.0, 6.5]
    psis = [0.0, np.pi / 2]
    filts = []
    for lam in lams:
        for th in thetas:
            for psi in psis:
                filts.append(gabor_kernel(k, th, lam, psi, 0.56 * lam, 0.5))
    filts = np.stack(filts[:out_ch])                      # [64,7,7]
    W = np.repeat(filts[:, None, :, :], in_ch, axis=1) / np.sqrt(in_ch)  # [64,3,7,7]
    return torch.tensor(W, dtype=torch.float32)


def build_model(front_end, trainable, seed):
    set_seed(seed)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    set_seed(seed)
    model.fc = nn.Linear(model.fc.in_features, 6)
    if front_end == "gabor":
        with torch.no_grad():
            model.conv1.weight.copy_(make_gabor_bank(model.conv1.out_channels,
                                                     model.conv1.kernel_size[0],
                                                     model.conv1.in_channels))
    tset = set(trainable) | {"bn1", "fc"}
    for name, p in model.named_parameters():
        stage = name.split(".")[0]
        p.requires_grad = (stage != "conv1") and (stage in tset)
    return model, tset


def enforce_bn_eval(model, tset):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            if name.split(".")[0] not in tset:
                mod.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def train_one(front_end, level, seed, loaders, args, device):
    train_loader, val_loader, test_loader = loaders
    model, tset = build_model(front_end, LEVELS[level], seed)
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hist = {"val_acc": [], "val_loss": []}
    best_vl, no_imp, best_state, epochs = float("inf"), 0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        epochs = epoch
        model.train(); enforce_bn_eval(model, tset)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        model.eval(); vl = vc = vn = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x); vl += crit(out, y).item() * y.size(0)
                vc += (out.argmax(1) == y).sum().item(); vn += y.size(0)
        vl /= vn; va = vc / vn
        hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"  [{front_end}/{level} s{seed}] ep{epoch:02d} vl {vl:.4f} va {va*100:.2f}%", flush=True)
        if vl < best_vl:
            best_vl, no_imp = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience:
                break
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    return {"front_end": front_end, "level": level, "seed": seed,
            "trainable_params": int(n_train), "epochs_trained": epochs,
            "best_val_loss": float(best_vl), "best_val_acc": float(max(hist["val_acc"])),
            "test_acc": float(test_acc), "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, classes = get_dataset_splits(
        DATA, batch_size=args.batch_size, num_workers=8, seed=42)
    loaders = (train_loader, val_loader, test_loader)

    print(f"device={device} seeds={args.seeds}", flush=True)
    results, t0 = [], time.time()
    for seed in args.seeds:
        for front_end in ["learned", "gabor"]:
            for level in LEVELS:
                rt = time.time()
                r = train_one(front_end, level, seed, loaders, args, device)
                r["duration_s"] = round(time.time() - rt, 1)
                results.append(r)
                print(f"DONE {front_end}/{level} s{seed}: test {r['test_acc']*100:.2f}% "
                      f"({r['trainable_params']:,} params, {r['epochs_trained']} ep, {r['duration_s']}s)", flush=True)
                json.dump({"config": vars(args), "results": results},
                          open(os.path.join(OUT, "gabor_ablation_results.json"), "w"), indent=2)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
