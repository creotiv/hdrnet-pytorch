import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class HDRDataset(Dataset):
    def __init__(self, image_path, params=None, suffix='', aug=False):
        self.image_path = image_path
        self.suffix = suffix
        self.aug = aug
        self.in_files = self.list_files(os.path.join(image_path, 'input'+suffix))
        ls = params['net_input_size']
        fs = params['net_output_size']
        self.low = transforms.Compose([
            transforms.Resize((ls,ls), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.correction = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0),
        ])
        self.out = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.full = transforms.Compose([
            transforms.Resize((fs,fs), Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, idx):
        fname = os.path.split(self.in_files[idx])[-1]
        imagein = Image.open(self.in_files[idx]).convert('RGB')
        imageout = Image.open(os.path.join(self.image_path, 'output'+self.suffix, fname)).convert('RGB')
        if self.aug:
            imagein = self.correction(imagein)
        imagein_low = self.low(imagein)
        imagein_full = self.full(imagein)
        imageout = self.out(imageout)

        return imagein_low,imagein_full,imageout

    def list_files(self, in_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files