import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def build_transforms(img_size:int=224, train:bool=True):
    base = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((img_size, img_size), antialias=True),
    ]
    if train:
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            T.RandomErasing(p=0.1),
        ]
    else:
        aug = []
    norm = [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base + aug + norm)
