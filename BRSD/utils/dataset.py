import albumentations as A
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BinarySegDatasetA(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]


        image = image.transpose(1, 2, 0)  # H,W,C
        mask = mask.transpose(1, 2, 0)    # H,W,C

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        if image.max() > 1:
            image = image / 255.0

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        image = image.transpose(2, 0, 1)  # C,H,W
        mask = mask.transpose(2, 0, 1)    # C,H,W

        return torch.from_numpy(image), torch.from_numpy(mask)


class BinarySegDatasetB(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]   # binary-seg dataset: (H,W,C)
        mask  = self.masks[idx]   # binary-seg dataset: (H,W)

       
        image = image.astype(np.float32)
        mask  = mask.astype(np.float32)

        if image.max() > 1:
            image = image / 255.0

        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        image = image.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)

        if mask.ndim == 2:
            mask = mask[..., np.newaxis]  # (H,W) → (H,W,1)
        mask = mask.transpose(2, 0, 1)    # (H,W,1) → (1,H,W)

        return torch.from_numpy(image), torch.from_numpy(mask)


class BraTS2020Dataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(224, 224)):
    
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.npy")))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       
        image = np.load(self.image_paths[idx])  
        mask = np.load(self.mask_paths[idx])    
        
        image = image.transpose(1, 2, 0)  # (240, 240, 4)

        
        resize_transform = A.Resize(self.target_size[0], self.target_size[1])
        image = resize_transform(image=image, mask=mask)
        image, mask = image['image'], image['mask']

      
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

       
        image = image.transpose(2, 0, 1)  # (4, H, W)

        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()