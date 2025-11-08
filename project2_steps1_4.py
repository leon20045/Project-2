
"""
@author: leon.olejarski
"""


#  AER850 Project 2 | Steps 1–4 (Train only — NO test here)
#  Two Custom DCNNs:
#    - custom_mobilenet  
#    - custom_vgg       
#  Saves:
#    best_custom_mobilenet.pt, best_custom_vgg.pt
#    hist_custom_mobilenet.pt, hist_custom_vgg.pt


# STEP 1: DATA PREP

from pathlib import Path
import json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms


ROOT_DIR = Path(r"C:\Users\leon2\Downloads\project2")


IMG_SIZE     = 500
BATCH_SIZE   = 32
NUM_WORKERS  = 0  

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR   = ROOT_DIR / "data" / "valid"
TEST_DIR  = ROOT_DIR / "data" / "test"     # used in Step 5

# Full-res
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1), shear=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, value='random'),
])


eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = ImageFolder(str(TRAIN_DIR), transform=train_tfms)
val_ds   = ImageFolder(str(VAL_DIR),   transform=eval_tfms)
test_ds  = ImageFolder(str(TEST_DIR),  transform=eval_tfms)  

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print("Classes:", train_ds.classes)
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")


# STEP 2: ARCH DEFINITIONS

import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Utility: Kaiming init for LeakyReLU layers
def kaiming_init(module, a=0.1):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

NUM_CLASSES = len(train_ds.classes)

# Custom DCNN #1: 
class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, g=1):
        pad = k // 2
        super().__init__(
            nn.Conv2d(cin, cout, k, s, pad, groups=g, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        mid = int(cin * expand)
        self.use_res = (stride == 1 and cin == cout)
        layers = []
        if expand != 1.0:
            layers.append(ConvBNAct(cin, mid, k=1, s=1))
        # depthwise
        layers.append(ConvBNAct(mid, mid, k=3, s=stride, g=mid))
        # project
        layers.append(nn.Conv2d(mid, cout, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(cout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y

class CustomMobileNetLite(nn.Module):
    def __init__(self, num_classes=3, width=1.0, dropout=0.3):
        super().__init__()
        c = lambda ch: max(8, int(ch * width))

        self.stem = ConvBNAct(3, c(32), k=3, s=2)     
        cfg = [
           
            (1.0, 16, 1, 1),
            (6.0, 24, 2, 2),  
            (6.0, 32, 3, 2),  
            (6.0, 64, 4, 2),  
            (6.0, 96, 3, 1),
            (6.0,160, 3, 2),  
            (6.0,192, 2, 1),
        ]
        in_ch = c(32)
        stages = []
        for t, ch, n, s in cfg:
            out_ch = c(ch)
            for i in range(n):
                stride = s if i == 0 else 1
                stages.append(InvertedResidual(in_ch, out_ch, stride, expand=t))
                in_ch = out_ch
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, c(256), k=1, s=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c(256), num_classes)
        )
        self.apply(lambda m: kaiming_init(m, a=0.1))

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x

def build_custom_mobilenet():
    return CustomMobileNetLite(NUM_CLASSES, width=1.0, dropout=0.35).to(device).to(memory_format=torch.channels_last)

# Custom DCNN #2: VGG-style (Small) 
def vgg_block(cin, cout, n_conv):
    layers = []
    for i in range(n_conv):
        layers += [
            nn.Conv2d(cin if i==0 else cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        ]
    layers.append(nn.MaxPool2d(2))  # downsample
    return nn.Sequential(*layers)

class CustomVGGSmall(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,   32, 2),   
            vgg_block(32,  64, 2),   
            vgg_block(64, 128, 2),   
            vgg_block(128,256, 2),   
            vgg_block(256,384, 2),   
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.apply(lambda m: kaiming_init(m, a=0.1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_custom_vgg():
    return CustomVGGSmall(NUM_CLASSES, dropout=0.45).to(device).to(memory_format=torch.channels_last)

print("Step 2 models ready (custom_mobilenet, custom_vgg).")


# STEP 3: FAST JOINT TUNING

from collections import defaultdict
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

TRIAL_EPOCHS    = 5
EARLY_PATIENCE  = 2
WEIGHT_DECAY    = 1e-4
MOMENTUM_SGD    = 0.9

IMG_SIZE_TRIAL  = 256
TRIAL_BATCH     = 64

trial_train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE_TRIAL, IMG_SIZE_TRIAL)),
    transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.05), scale=(0.9,1.1), shear=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.25, value='random'),
])
trial_val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE_TRIAL, IMG_SIZE_TRIAL)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

trial_train_full = ImageFolder(str(TRAIN_DIR), transform=trial_train_tfms)
trial_val_full   = ImageFolder(str(VAL_DIR),   transform=trial_val_tfms)

def stratified_subset(dataset: ImageFolder, max_per_class=150, seed=42):
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(list)
    for idx, (_, y) in enumerate(dataset.samples):
        by_cls[y].append(idx)
    keep = []
    for y, idxs in by_cls.items():
        idxs = np.array(idxs)
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False)
        keep.extend(idxs.tolist())
    keep.sort()
    return Subset(dataset, keep)

trial_train_ds = stratified_subset(trial_train_full, max_per_class=150)
trial_val_ds   = trial_val_full

TRAIN_LOADER = DataLoader(trial_train_ds, batch_size=TRIAL_BATCH, shuffle=True,
                          num_workers=0, pin_memory=True)
VAL_LOADER   = DataLoader(trial_val_ds,   batch_size=TRIAL_BATCH, shuffle=False,
                          num_workers=0, pin_memory=True)

def eval_once(model, loader, criterion):
    model.eval()
    tot_loss = tot_ok = tot_n = 0
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
    with amp, torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * xb.size(0)
            tot_ok   += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
    return tot_loss / tot_n, tot_ok / tot_n

def count_params(m):
    tot = sum(p.numel() for p in m.parameters())
    trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return tot, trn

def build_model(arch):
    if arch == "custom_mobilenet":
        m = build_custom_mobilenet()
    elif arch == "custom_vgg":
        m = build_custom_vgg()
    else:
        raise ValueError("arch must be 'custom_mobilenet' or 'custom_vgg'")
    return m

def train_quick_trial(arch, opt_name, lr, dropout=None, epochs=TRIAL_EPOCHS):
   
    model = build_model(arch)
    tot, trn = count_params(model)
    print(f"[build] {arch} | total={tot/1e6:.2f}M trainable={trn/1e6:.2f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM_SGD,
                                    weight_decay=WEIGHT_DECAY, nesterov=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_acc, best_state, no_improve = 0.0, None, 0
    hist = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        pbar = tqdm(total=len(TRAIN_LOADER), leave=False,
                    desc=f"{arch} {opt_name} lr={lr:.1e} ep {ep}/{epochs}")
        for xb, yb in TRAIN_LOADER:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
            tr_n    += xb.size(0)
            pbar.update(1)
        pbar.close()

        tr_loss /= tr_n
        tr_acc   = tr_ok / tr_n
        val_loss, val_acc = eval_once(model, VAL_LOADER, criterion)

        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(val_loss);   hist["val_acc"].append(val_acc)

        print(f"[trial {arch}] Ep{ep:02d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.3f}  val {val_loss:.4f}/{val_acc:.3f}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state, no_improve = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE:
                print(f"  ↳ early stop (no val acc improvement {EARLY_PATIENCE} epochs)")
                break
    return best_val_acc, hist, best_state

# Small grid for both models
trial_space = {
    "arch":    ["custom_mobilenet", "custom_vgg"],
    "opt":     ["adam", "sgd"],
    "lr":      [1e-3, 7e-4, 3e-4],
}

leaderboard = []
for arch in trial_space["arch"]:
    for optn in trial_space["opt"]:
        for lr in trial_space["lr"]:
            best_acc, hist, state = train_quick_trial(arch, optn, lr)
            cfg = {"arch": arch, "opt": optn, "lr": lr}
            leaderboard.append((best_acc, cfg, state))
            print(f"--> DONE | best val_acc={best_acc:.4f} | cfg={cfg}")

leaderboard.sort(key=lambda x: x[0], reverse=True)
print("\n===== LEADERBOARD (top → bottom) =====")
for i,(acc,cfg,_) in enumerate(leaderboard,1):
    print(f"{i:2d}. {acc:.4f} | {cfg}")

def best_for_arch(arch):
    for acc,cfg,state in leaderboard:
        if cfg["arch"]==arch: return acc,cfg,state
    return None,None,None

best_mobile_acc,best_mobile_cfg,best_mobile_state = best_for_arch("custom_mobilenet")
best_vgg_acc,   best_vgg_cfg,  best_vgg_state    = best_for_arch("custom_vgg")

print("\nSelected CustomMobileNet:",best_mobile_cfg,"≈",f"{best_mobile_acc:.4f}")
print("Selected CustomVGG:",best_vgg_cfg,"≈",f"{best_vgg_acc:.4f}")


# STEP 4: FULL TRAIN

from tqdm.auto import tqdm

FULL_EPOCHS     = 45
EARLY_PATIENCE4 = 4
MOMENTUM_SGD    = 0.9
WEIGHT_DECAY    = 1e-4

def make_optimizer(params, opt_name, lr):
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM_SGD,
                               weight_decay=WEIGHT_DECAY, nesterov=True)

def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = tot_ok = tot_n = 0
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()), torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * xb.size(0)
            tot_ok   += (logits.argmax(1) == yb).sum().item()
            tot_n    += xb.size(0)
    return tot_loss / tot_n, tot_ok / tot_n

def rebuild_for_full(cfg):
    if cfg["arch"] == "custom_mobilenet":
        model = build_custom_mobilenet()
    else:
        model = build_custom_vgg()
    return model

def train_full(arch_name, cfg, save_path, force_batch=None):
    batch_size = force_batch if force_batch else BATCH_SIZE
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    vl_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    model = rebuild_for_full(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, cfg["opt"], cfg["lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss, best_val_acc = float("inf"), 0.0
    best_state = None
    hist = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "lr":[]}
    no_improve = 0

    for ep in range(1, FULL_EPOCHS+1):
        model.train()
        tr_loss = tr_ok = tr_n = 0
        pbar = tqdm(total=len(tr_loader), leave=False, desc=f"{arch_name} | ep {ep}/{FULL_EPOCHS}")
        for xb, yb in tr_loader:
            xb = xb.to(device).to(memory_format=torch.channels_last)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * xb.size(0)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
            tr_n    += xb.size(0)
            pbar.update(1)
        pbar.close()

        tr_loss /= tr_n
        tr_acc   = tr_ok / tr_n

        val_loss, val_acc = evaluate(model, vl_loader, criterion)
        scheduler.step()

        last_lr = optimizer.param_groups[0]["lr"]
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(val_loss);   hist["val_acc"].append(val_acc)
        hist["lr"].append(last_lr)

        print(f"[{arch_name}] Epoch {ep:02d}/{FULL_EPOCHS} | "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  val {val_loss:.4f}/{val_acc:.3f}  lr {last_lr:.2e}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"cfg": cfg, "classes": train_ds.classes,
                        "model_state": best_state}, save_path)
            torch.save(hist, save_path.replace(".pt", "_hist.pt"))
        else:
            no_improve += 1
            if no_improve >= EARLY_PATIENCE4:
                print(f"  ↳ early stop (no val loss improvement for {EARLY_PATIENCE4} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val_loss, best_val_acc

print("\n=== Step 4: Train CustomMobileNet @500px ===")
mobile_model, mobile_hist, mobile_vloss, mobile_vacc = train_full(
    "custom_mobilenet", best_mobile_cfg, save_path="best_custom_mobilenet.pt", force_batch=BATCH_SIZE
)

print("\n=== Step 4: Train CustomVGG @500px ===")
vgg_model, vgg_hist, vgg_vloss, vgg_vacc = train_full(
    "custom_vgg", best_vgg_cfg, save_path="best_custom_vgg.pt", force_batch=BATCH_SIZE
)

torch.save(mobile_hist, "hist_custom_mobilenet.pt")
torch.save(vgg_hist,    "hist_custom_vgg.pt")

print("\nStep 4 complete — checkpoints and histories saved:")
print("  - best_custom_mobilenet.pt, hist_custom_mobilenet.pt")
print("  - best_custom_vgg.pt,       hist_custom_vgg.pt")

# Figure A: training curves
import matplotlib.pyplot as plt
plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.plot(mobile_hist["train_loss"], label="MobileNetLite train")
plt.plot(mobile_hist["val_loss"],   label="MobileNetLite val")
plt.plot(vgg_hist["train_loss"],    label="VGGSmall train", linestyle="--")
plt.plot(vgg_hist["val_loss"],      label="VGGSmall val",   linestyle="--")
plt.title("Loss vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(mobile_hist["train_acc"], label="MobileNetLite train")
plt.plot(mobile_hist["val_acc"],   label="MobileNetLite val")
plt.plot(vgg_hist["train_acc"],    label="VGGSmall train", linestyle="--")
plt.plot(vgg_hist["val_acc"],      label="VGGSmall val",   linestyle="--")
plt.title("Accuracy vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.tight_layout()
plt.savefig("figure_model_performance.png", dpi=160)
print("Saved: figure_model_performance.png")