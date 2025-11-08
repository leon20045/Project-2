
"""
AER850 Project 2 | Step 5 — Model Testing (CPU only)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt



ROOT_DIR = Path(r"C:\Users\leon2\Downloads\project2\Project 2 Data") 
MODEL_PATH = Path("best_custom_vgg.pt")
SAVE_FIG = "figure_test_custom_vgg.png"
DEVICE = torch.device("cpu")   # CPU only
IMG_SIZE = 500


eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# Rebuild the model
import torch.nn as nn

def vgg_block(cin, cout, n_conv):
    layers = []
    for i in range(n_conv):
        layers += [
            nn.Conv2d(cin if i == 0 else cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True)
        ]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CustomVGGSmall(nn.Module):
    def __init__(self, num_classes=3, dropout=0.45):
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(3,   32, 2),
            vgg_block(32,  64, 2),
            vgg_block(64, 128, 2),
            vgg_block(128,256,2),
            vgg_block(256,384,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#checkpoint 
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint.get("classes", ["crack","missing-head","paint-off"])
state_dict = checkpoint["model_state"]
cfg = checkpoint["cfg"]
print("[Info] Loaded model config:", cfg)
print("[Info] Classes:", classes)

# Rebuild model & load weights
model = CustomVGGSmall(num_classes=len(classes))
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# test image prep
TEST_DIR = ROOT_DIR / "data" / "test"
test_samples = [
    TEST_DIR / "crack" / "test_crack.jpg",
    TEST_DIR / "missing-head" / "test_missinghead.jpg",
    TEST_DIR / "paint-off" / "test_paintoff.jpg",
]

images, labels, preds, probs = [], [], [], []

for img_path in test_samples:
    img = Image.open(img_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred_idx = prob.argmax(1).item()
        conf = prob[0, pred_idx].item()

    images.append(img)
    labels.append(img_path.parent.name)
    preds.append(classes[pred_idx])
    probs.append(conf)
    print(f"{img_path.name}: predicted '{classes[pred_idx]}' ({conf*100:.1f}%)")

#Visualization 
plt.figure(figsize=(11,4))
for i, (img, label, pred, conf) in enumerate(zip(images, labels, preds, probs)):
    plt.subplot(1,3,i+1)
    plt.imshow(img)
    title = f"True: {label}\nPred: {pred} ({conf*100:.1f}%)"
    color = "green" if pred == label else "red"
    plt.title(title, color=color, fontsize=10)
    plt.axis("off")

plt.suptitle("Step 5 — Model Testing Results (Custom VGG)", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=200)
print(f"[Saved] {SAVE_FIG}")
plt.show()
