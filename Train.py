from NodeConvs_Net import NodeConvs_Net
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = np.load('bloodmnist_70_15_15.npz')
print("Available arrays:", data.files)

CLASS_NAMES =  [
    "basophil", "eosinophil", "erythroblast", "immature granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

splits = {"train" : (data['train_images'],  data['train_labels'].reshape(-1)),
        "val" : (data['val_images'], data['val_labels'].reshape(-1)),
        "test" : (data['test_images'], data['test_labels'].reshape(-1))}
print(len(splits["train"][1]))
