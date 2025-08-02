"""
BRSET DataLoader (based on official implementation)
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import transforms

class BRSETDataset(Dataset):
    """Official BRSET Dataset implementation"""
    
    def __init__(self, df, images_path, transform=None, task='multi_label'):
        self.df = df.reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform
        self.task = task
        
        # Define label columns based on task
        if task == 'diabetes':
            self.label_cols = ['diabetes']
        elif task == 'DR_2class':
            self.label_cols = ['diabetic_retinopathy']
        elif task == 'DR_3class':
            self.label_cols = ['DR_ICDR']
        elif task == 'sex':
            self.label_cols = ['patient_sex']
        else:  # multi_label
            self.label_cols = [
                'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
                'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 
                'hemorrhage', 'retinal_detachment', 'myopic_fundus', 
                'increased_cup_disc', 'other'
            ]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_name = row['image_id']
        image_path = os.path.join(self.images_path, image_name)
        
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create mock fundus-like image for development
            image = self.create_mock_fundus_image()
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        if self.task == 'sex':
            # Convert to 0/1 (original is 1/2)
            labels = torch.tensor(row[self.label_cols[0]] - 1, dtype=torch.long)
        elif self.task == 'DR_3class':
            # Multi-class classification
            labels = torch.tensor(row[self.label_cols[0]], dtype=torch.long)
        else:
            # Binary or multi-label
            labels = torch.tensor([row[col] for col in self.label_cols], dtype=torch.float32)
            if len(labels) == 1:
                labels = labels[0]  # Single label for binary tasks
        
        # Return patient metadata
        metadata = {
            'age': row['patient_age'],
            'sex': row['patient_sex'],
            'diabetes_time': row.get('diabetes_time', 0),
            'insulin_use': row.get('insulin_use', 0)
        }
        
        return image, labels, metadata
    
    def create_mock_fundus_image(self):
        """Create realistic fundus-like mock image"""
        # Create fundus-like circular image with reddish background
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Create circular mask
        center = (112, 112)
        radius = 100
        y, x = np.ogrid[:224, :224]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Fundus-like colors (reddish background)
        img[mask] = [120 + np.random.randint(-20, 20), 
                     60 + np.random.randint(-15, 15), 
                     40 + np.random.randint(-10, 10)]
        
        # Add some vessel-like structures
        for _ in range(3):
            start_x, start_y = np.random.randint(50, 174, 2)
            end_x, end_y = np.random.randint(50, 174, 2)
            thickness = np.random.randint(1, 3)
            # Simple line drawing
            if abs(end_x - start_x) > abs(end_y - start_y):
                for x in range(min(start_x, end_x), max(start_x, end_x)):
                    y = start_y + (end_y - start_y) * (x - start_x) // (end_x - start_x)
                    if 0 <= y < 224:
                        img[max(0, y-thickness):min(224, y+thickness), x] = [80, 30, 20]
        
        return Image.fromarray(img)

def process_transforms(shape=(224, 224), augment=True):
    """Get transforms for BRSET (official approach)"""
    if augment:
        return transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
