import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BloodMNISTDataset(Dataset):
 
    def __init__(self, images, labels, mean, std, augment=False):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.mean = mean
        self.std = std
        self.augment = augment
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # -> [0, 1]
 
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()  # horizontal flip
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy()  # vertical flip
 
        img = (img - self.mean) / self.std                     # normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float()   # HWC -> CHW
 
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
 
 
def get_dataloaders(npz_path="BloodMNIST/bloodmnist_70_15_15.npz", batch_size=64, num_workers=2):
    data = np.load(npz_path)
 
    train_images, train_labels = data["train_images"], data["train_labels"].reshape(-1)
    val_images, val_labels = data["val_images"], data["val_labels"].reshape(-1)
    test_images, test_labels = data["test_images"], data["test_labels"].reshape(-1)
 
    train_float = train_images.astype(np.float32) / 255.0
    mean = train_float.reshape(-1, train_float.shape[-1]).mean(axis=0)
    std = train_float.reshape(-1, train_float.shape[-1]).std(axis=0)
 
    train_ds = BloodMNISTDataset(train_images, train_labels, mean, std, augment=True)
    val_ds = BloodMNISTDataset(val_images, val_labels, mean, std, augment=False)
    test_ds = BloodMNISTDataset(test_images, test_labels, mean, std, augment=False)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
 
    return train_loader, val_loader, test_loader, (mean, std)
 
 



