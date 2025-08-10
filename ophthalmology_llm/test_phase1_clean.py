import argparse, os
import torch
from PIL import Image
import torchvision.transforms as T
from src.model import create_model
from llm.clinical_llm import create_clinical_llm

LABELS = [
    "diabetic_retinopathy",
    "macular_edema",
    "amd",
    "retinal_detachment",
    "increased_cup_disc",
    "other",
]

def load_image(path):
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=str, help="Path to a fundus image (.jpg)")
    ap.add_argument("--weights", default="models/phase1_best_model.pth", type=str)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    model = create_model()
    if os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print("✅ Loaded weights:", args.weights)
    else:
        print("⚠️ Weights not found, using randomly initialized model.")

    model.eval()
    x = load_image(args.image)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    print("\n📊 Probabilities:")
    for k, p in zip(LABELS, probs):
        print(f"  {k}: {p:.3f}")

    # simple patient
    llm = create_clinical_llm()
    patient = {"age": 67, "sex": "F"}
    print("\n" + llm.summarize(patient, probs))

if __name__ == "__main__":
    main()
