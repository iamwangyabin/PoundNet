import os
import io
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A

from datasets import load_from_disk

def check_transform_lib(transform):
    module_name = transform.__class__.__module__
    if module_name.startswith('albumentations'):
        return 'albumentations'
    elif module_name.startswith('torchvision'):
        return 'torchvision'
    else:
        return 'unknown'

class ArrowDatasets(Dataset):
    def __init__(self, data_root, trsf, subset, split='train'):
        self.dataroot = data_root
        self.dataset = load_from_disk(data_root)

        self.split = split
        self.image_pathes = []
        self.labels = []

        json_file = os.path.join(self.dataroot, f'{split}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        for img_rel_path, label in data[subset].items():
            img_full_path = os.path.join(self.dataroot, img_rel_path)
            self.image_pathes.append(img_full_path)
            self.labels.append(label)

        with open(os.path.join(self.dataroot, 'mapping.json'), 'r') as f:
            mapping = json.load(f)
        self.mapping = {}
        for path, idx in mapping.items():
            img_full_path = os.path.join(self.dataroot, path)
            self.mapping[img_full_path] = idx


        for idx, transform in enumerate(trsf):
            self.lib = check_transform_lib(transform)

        if self.lib == 'albumentations':
            self.transform_chain = A.Compose(trsf)
        elif self.lib == 'torchvision':
            self.transform_chain = transforms.Compose(trsf)

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        img_path = self.image_pathes[idx]
        index = self.mapping[img_path]
        example = self.dataset[index]
        image = Image.open(io.BytesIO(example['image'])).convert('RGB')
        label = self.labels[idx]
        if self.lib == 'albumentations':
            image = np.array(image)
            image = self.transform_chain(image=image)["image"].float()
        elif self.lib == 'torchvision':
            image = self.transform_chain(image)
        return image, label



