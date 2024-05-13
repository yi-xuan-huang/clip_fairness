import logging
import os
from tqdm import tqdm
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class MIMIC(Dataset):
    def __init__(self, data, image_path, transform=None):
        self.data = data
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        d = self.data[idx]
        report =d['text']
        image = Image.open(os.path.join(self.image_path, d['image'])).convert('RGB')

        if self.transform:
            image = self.transform(image) 

        return {
            'image': image,
            'text': report, 
        }  

class MIMIC_Labels(Dataset):
    def __init__(self, data, image_path, transform=None):
        self.data = data
        self.image_path = image_path
        self.transform = transform
        # self.data = [d for d in data if d['view'] == 'PA']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        d = self.data[idx]
        d['name'] = d['image']
        image = Image.open(os.path.join(self.image_path, d['image'])).convert('RGB')

        if self.transform:
            image = self.transform(image) 

        d['image'] = image
        return d

class MIMIC_Demo(Dataset):
    def __init__(self, data, image_path, transform=None):
        self.data = data
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        d = self.data[idx]
        d['name'] = d['image']
        image = Image.open(os.path.join(self.image_path, d['image'])).convert('RGB')

        if self.transform:
            image = self.transform(image) 

        d['image'] = image
        return d
