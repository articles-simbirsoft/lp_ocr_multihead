import glob
import json
import os

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset


class NpOcrDataset(Dataset):
    def __init__(self, data_path, transform, label_encoder):
        super().__init__()
        self.data_path = data_path
        self.image_fnames = glob.glob(os.path.join(data_path, "img", "*.png"))
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        img_fname = self.image_fnames[idx]
        img = cv2.imread(img_fname)
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        img = img.transpose(2, 0, 1)
        
        label_fname = os.path.join(self.data_path, "ann",
                                   os.path.basename(img_fname).replace(".png", ".json"))
        with open(label_fname, "rt") as label_file:
            label_struct = json.load(label_file)
            label = label_struct["description"]
        label = self.label_encoder.encode(label)

        return img, [c for c in label]

class LabelEncoder():
    def __init__(self, max_len):
        self.alphabet = "-1234567890ABEKMHOPCTYX"
        self.max_len = max_len
    
    def encode(self, label):
        if len(label) > self.max_len:
            label = label[:self.max_len]
        elif len(label) < self.max_len:
            label += self.alphabet[0] * (self.max_len - len(label))
        
        label_encoded = [self.alphabet.index(c) if c in self.alphabet else 0
                         for c in label]
        return label_encoded
    
    def decode(self, label):
        label_decoded = [
            self.alphabet[c] if 0 <= c < len(self.alphabet) else self.alphabet[0]
            for c in label
        ]
        return label_decoded

def get_train_transform(processing_size):
    return A.Compose([
        A.Normalize(),
        A.Resize(*processing_size[::-1])
    ])

def get_valid_transform(processing_size):
    return get_train_transform(processing_size)
