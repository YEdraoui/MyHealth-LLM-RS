import os, argparse, numpy as np
from PIL import Image
import torch as T
import torch.nn.functional as F
from torchvision.transforms import v2 as T2
from src.utils import read_config, get_device
from src.dataset_brset import BRSETDataset
from src.vision_backbone import build_model

ap=argparse.ArgumentParser()
ap.add_argument("--split", choices=["train","val","test"], default="test")
ap.add_argument("--index", type=int, default=0)
ap.add_argument("--cls", default="diabetic_retinopathy")
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--ckpt", default="models/phase2_best.pt")
ap.add_argument("--out", default="reports/gradcam/example.png")
ap.add_argument("--backbone", default="resnet18")
args=ap.parse_args()

conf=read_config(); labels=conf["labels"]; ncls=len(labels); jcls=labels.index(args.cls)
dev=get_device()

# Load dataset row
import csv
split_csv=os.path.join(conf["splits_dir"], f"{args.split}.csv")
with open(split_csv, newline="") as f: rows=list(csv.DictReader(f))
row=rows[args.index]; img_path=os.path.join(conf["data_root"], "fundus_photos", row["filename"])

# Model
model=build_model(ncls, backbone=args.backbone, pretrained=False).to(dev)
sd=T.load(args.ckpt, map_location="cpu"); model.load_state_dict(sd, strict=False)
model.eval()

# pick a conv layer for resnet18
target_layer = None
if args.backbone.lower().startswith("resnet"):
    target_layer = model.layer4[-1].conv2
else:
    # naive fallback: try to find last conv-like module
    for m in reversed(list(model.modules())):
        if isinstance(m, T.nn.Conv2d):
            target_layer = m; break
if target_layer is None:
    raise SystemExit("Could not find a conv layer to hook for Grad-CAM.")

activations = []
gradients = []
def fwd_hook(m, i, o): activations.append(o.detach())
def bwd_hook(m, gi, go): gradients.append(go[0].detach())

h1=target_layer.register_forward_hook(fwd_hook)
h2=target_layer.register_full_backward_hook(bwd_hook)

# Preprocess
tf = T2.Compose([T2.ToImage(), T2.ToDtype(T.float32, scale=True),
                 T2.Resize((args.img_size,args.img_size), antialias=True),
                 T2.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0).to(dev)

# Forward + backward on target class logit
x.requires_grad_(True)
logits = model(x)
score = logits[0, jcls]
model.zero_grad(set_to_none=True)
score.backward()

A = activations[-1]           # [1, C, H, W]
G = gradients[-1]             # [1, C, H, W]
weights = G.mean(dim=(2,3), keepdim=True)   # GAP over grad
cam = (A*weights).sum(dim=1, keepdim=True)  # [1,1,H,W]
cam = F.relu(cam)
cam = F.interpolate(cam, size=(args.img_size,args.img_size), mode="bilinear", align_corners=False)
cam = cam[0,0].cpu().numpy()
cam = (cam - cam.min()) / (cam.max() + 1e-8)

# overlay
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.imshow(np.array(img.resize((args.img_size,args.img_size))))
plt.imshow(cam, alpha=0.35)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.axis("off"); plt.tight_layout(); plt.savefig(args.out, dpi=180, bbox_inches="tight")
print("Saved", args.out)

h1.remove(); h2.remove()
