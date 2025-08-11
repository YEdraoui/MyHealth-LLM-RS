import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class MultimodalFundusDataset(Dataset):
    def __init__(self, csv_file, img_dir, split='train'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.tabular_cols = ['patient_age','sex','diabetes','hypertension']
        self.label_cols = ['diabetic_retinopathy','macular_edema','amd',
                           'retinal_detachment','increased_cup_disc','other']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tabular = torch.tensor(row[self.tabular_cols].values, dtype=torch.float32)
        labels = torch.tensor(row[self.label_cols].values, dtype=torch.float32)
        return image, tabular, labels
