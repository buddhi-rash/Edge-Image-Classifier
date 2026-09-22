import numpy as np
import matplotlib.pyplot as plt

data = np.load('bloodmnist_70_15_15.npz')
print("Available arrays:", data.files)

CLASS_NAMES =  [
    "basophil", "eosinophil", "erythroblast", "immature granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

train_imgs = data['train_images']
val_imgs = data['val_images']
test_imgs = data['test_images']
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']

splits = {"train" : (train_imgs, train_labels.reshape(-1)),
        "val" : (val_imgs, val_labels.reshape(-1)),
        "test" : (test_imgs, test_labels.reshape(-1))}

#image details analysis
image_shape = train_imgs.shape[1:]
print(f'size and shape of the loaded images is {image_shape}')
print(f"dtype={train_imgs.dtype}  pixel range=[{train_imgs.min()}, {train_imgs.max()}]")

#check for NaNs or Blank images
for name, (imgs, labels) in splits.items():
    assert not np.isnan(imgs).any(), f"{name} contains NaNs!"
    per_image_std = imgs.reshape(imgs.shape[0], -1).astype(np.float32).std(axis=1)
    n_blank = int((per_image_std == 0).sum())
    if n_blank > 0:
        print(f"WARNING: {name} has {n_blank} blank/constant images")
print("No NaNs found; blank-image check complete.\n")


#check for class diversity and datapoint inbalance
counts = {name: np.bincount(labels, minlength=8) for name, (_, labels) in splits.items()}
header = f"{'Class':<24}" + "".join(f"{s:>10}" for s in splits)
print(header)
for c in range(8):
    row = f"{CLASS_NAMES[c]:<24}" + "".join(f"{counts[s][c]:>10}" for s in splits)
    print(row)
 
train_counts = counts["train"]
imbalance_ratio = train_counts.max() / train_counts.min()
print(f"\nTrain-set class imbalance ratio (max/min class count): {imbalance_ratio:.2f}x")