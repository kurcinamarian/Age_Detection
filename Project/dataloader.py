import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, csv_path, root_dir, split='train', augment=False):
        
        df = pd.read_csv(csv_path)
        
        self.df = df[df['dataset'] == split].reset_index(drop=True)
        self.root_dir = root_dir
        self.augment = augment

        if augment:
            # Perform image augmentations
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # Get the image and label
        
        img_rel_path = self.df.iloc[idx]['ID']
        label = int(self.df.iloc[idx]['Class id'])
        
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
            
        return image, label

def create_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4):
    
    # Create datasets for train, valid, test
    train_dataset = ImageDataset(csv_path, root_dir, split='train', augment=False)
    valid_dataset = ImageDataset(csv_path, root_dir, split='valid', augment=False)
    test_dataset = ImageDataset(csv_path, root_dir, split='test', augment=False)
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

