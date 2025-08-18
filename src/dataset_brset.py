import os, csv
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class BRSETDataset(Dataset):
    def __init__(self, data_root:str, csv_rows:List[Dict[str,str]], labels:List[str], transform=None):
        self.data_root=data_root
        self.rows=csv_rows
        self.labels=labels
        self.transform=transform
        self.img_dir=os.path.join(data_root, "fundus_photos")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx:int):
        r=self.rows[idx]
        path=os.path.join(self.img_dir, r["filename"])
        img=Image.open(path).convert("RGB")
        if self.transform: img=self.transform(img)
        y=torch.tensor([1.0 if str(r.get(l,"0")).strip() in ("1","True","true") else 0.0 for l in self.labels], dtype=torch.float32)
        return img, y

def _read_csv(path:str):
    with open(path, newline="") as f: return list(csv.DictReader(f))

def make_loaders(data_root:str, splits_dir:str, labels:List[str], img_size:int=224,
                 batch_size:int=32, num_workers:int=4):
    from src.transforms import build_transforms
    tf_train=build_transforms(img_size, train=True)
    tf_eval =build_transforms(img_size, train=False)
    def dl(name, tfm, shuffle):
        rows=_read_csv(os.path.join(splits_dir, f"{name}.csv"))
        ds=BRSETDataset(data_root, rows, labels, tfm)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=False,
                          persistent_workers=(num_workers>0))
    loaders={}
    if os.path.isfile(os.path.join(splits_dir,"train.csv")): loaders["train"]=dl("train", tf_train, True)
    if os.path.isfile(os.path.join(splits_dir,"val.csv")):   loaders["val"]  =dl("val",   tf_eval,  False)
    if os.path.isfile(os.path.join(splits_dir,"test.csv")):  loaders["test"] =dl("test",  tf_eval,  False)
    return loaders
