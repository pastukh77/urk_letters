import os
import glob
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


folder = "letters_cleaned"
labels = os.listdir(folder)
labels.remove(".DS_Store")


labels_to_int = {labels[i] : i for i in range(len(labels))}
int_to_labels = {i : labels[i] for i in range(len(labels))}



class LettersDataset(Dataset):
    def __init__(self, image_dir, mode = 'train', transforms = None):
        super().__init__()
        self.images = glob.glob(f'{folder}/*/*.jpg')
        self.transforms = transforms
        self.mode = mode
    
    def __getitem__(self, index: int):
        image_path = self.images[index]
        img = Image.open(image_path).convert("L")
        label = image_path.split("/")[1]
        label_int = labels_to_int[label]
        if self.transforms:
            img = self.transforms(img)
        return img, label_int

    def __len__(self):
        return len(self.images)

transforms = T.Compose([
            T.Resize((36, 36)),
            T.ToTensor()
        ])

lettersDataset = LettersDataset(folder, transforms=transforms)

train_split = 0.8

indices = torch.randperm(len(lettersDataset)).tolist()
train_dataset = torch.utils.data.Subset(lettersDataset, indices[:int(len(lettersDataset) * 0.9)])
test_dataset = torch.utils.data.Subset(lettersDataset, indices[:int(len(lettersDataset) * 0.9):])


train_loader = DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True,
    num_workers=8
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8
)

