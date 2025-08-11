import argparse, torch
from PIL import Image
from torchvision import transforms
from src.models import MultimodalClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--metadata', type=float, nargs='+', required=True)
parser.add_argument('--weights', type=str, required=True)
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

image = Image.open(args.image).convert('RGB')
image = transform(image).unsqueeze(0).to(device)
tabular = torch.tensor(args.metadata, dtype=torch.float32).unsqueeze(0).to(device)

model = MultimodalClassifier().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
model.eval()

labels = ['diabetic_retinopathy','macular_edema','amd','retinal_detachment','increased_cup_disc','other']
with torch.no_grad():
    outputs = torch.sigmoid(model(image, tabular)).cpu().numpy()[0]
    for lbl, prob in zip(labels, outputs):
        print(f"{lbl}: {prob:.3f}")
