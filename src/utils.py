import os, random, json
import numpy as np
import torch as T
import yaml

def read_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); T.manual_seed(seed)
    if T.cuda.is_available(): T.cuda.manual_seed_all(seed)

def get_device(prefer="auto"):
    if prefer!="auto": return T.device(prefer)
    if T.cuda.is_available(): return T.device("cuda")
    if getattr(T.backends, "mps", None) and T.backends.mps.is_available():
        # Best-practice flags for MPS stability
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO","0.0")
        return T.device("mps")
    return T.device("cpu")

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)
