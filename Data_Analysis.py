import numpy as np
import matplotlib.pyplot as plt
# Quick dataset audit for image format, quality, and class balance.

# Load the saved dataset from the npz archive.
data = np.load('BloodMNIST/bloodmnist_70_15_15.npz')
# Print the available arrays in the file.
print("Available arrays:", data.files)

# Match names to class indices for readable output.
CLASS_NAMES =  [
    "basophil", "eosinophil", "erythroblast", "immature granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

# Extract train, validation, and test image arrays.
train_imgs = data['train_images']
val_imgs = data['val_images']
test_imgs = data['test_images']
# Extract the label arrays for each split.
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']

# Flatten labels to one dimension for counting.
splits = {"train" : (train_imgs, train_labels.reshape(-1)),
        "val" : (val_imgs, val_labels.reshape(-1)),
        "test" : (test_imgs, test_labels.reshape(-1))}

# Inspect image dimensions and pixel intensity range.
image_shape = train_imgs.shape[1:]
print(f'size and shape of the loaded images is {image_shape}')
print(f"dtype={train_imgs.dtype}  pixel range=[{train_imgs.min()}, {train_imgs.max()}]")

# Ensure no invalid or missing pixel values appear.
for name, (imgs, labels) in splits.items():
    assert not np.isnan(imgs).any(), f"{name} contains NaNs!"
    # Measure how uniform each image is to find blanks.
    per_image_std = imgs.reshape(imgs.shape[0], -1).astype(np.float32).std(axis=1)
    n_blank = int((per_image_std == 0).sum())
    if n_blank > 0:
        print(f"WARNING: {name} has {n_blank} blank/constant images")
print("No NaNs found; blank-image check complete.\n")

# Count samples per class for balance checks.
counts = {name: np.bincount(labels, minlength=8) for name, (_, labels) in splits.items()}
# Build a compact class-count table for each split.
header = f"{'Class':<24}" + "".join(f"{s:>10}" for s in splits)
print(header)
for c in range(8):
    row = f"{CLASS_NAMES[c]:<24}" + "".join(f"{counts[s][c]:>10}" for s in splits)
    print(row)
 
# Measure how imbalanced the training set is.
train_counts = counts["train"]
imbalance_ratio = train_counts.max() / train_counts.min()
print(f"\nTrain-set class imbalance ratio (max/min class count): {imbalance_ratio:.2f}x")

# Plot class counts across all splits.
fig, ax = plt.subplots(figsize=(10, 5))
# Use one bar group per split for comparison.
x = np.arange(8)
width = 0.25
for i, name in enumerate(splits):
    ax.bar(x + i * width, counts[name], width, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_ylabel("Number of images")
ax.set_title("BloodMNIST class distribution per split")
ax.legend()
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
plt.close(fig)
print("Saved class_distribution.png")

# Display representative samples from each class.
fig, axes = plt.subplots(8, 5,
                          figsize=(2 * 5, 2 * 8))
for c in range(8):
    # Pick the first five examples for each class.
    idxs = np.where(train_labels == c)[0][:5]
    for j in range(5):
        ax = axes[c, j]
        if j < len(idxs):
            ax.imshow(train_imgs[idxs[j]])
        ax.axis("off")
    # Add the class name to the first column.
    axes[c, 0].text(-0.3, 0.5, CLASS_NAMES[c], transform=axes[c, 0].transAxes,
                     fontsize=9, ha="right", va="center")
plt.tight_layout()
plt.savefig("sample_grid.png", dpi=150)
plt.show()
plt.close(fig)
print("Saved sample_grid.png")

# Compute the normalization values for the training data.
train_float = train_imgs.astype(np.float32) / 255.0
# Convert the image tensor to a 2D array for mean/std.
mean = train_float.reshape(-1, train_float.shape[-1]).mean(axis=0)
std = train_float.reshape(-1, train_float.shape[-1]).std(axis=0)
print("\nPer-channel mean (train, [0,1] scale):", mean)
print("Per-channel std  (train, [0,1] scale):", std)