from medmnist import BloodMNIST
import numpy as np
from sklearn.model_selection import train_test_split


Image_res = 64
random_seed = 21

train_set = BloodMNIST(split= "train", download= True, as_rgb = True, size = Image_res)
val_set = BloodMNIST(split= "val", download= True, as_rgb = True, size = Image_res)
test_set = BloodMNIST(split= "test", download= True, as_rgb = True, size = Image_res)

merged_img = np.concatenate([val_set.imgs, test_set.imgs], axis = 0)
merged_labels = np.concatenate([val_set.labels, test_set.labels], axis = 0)
merged_labels_flat = merged_labels.reshape(-1) 

val_imgs , test_imgs , val_labels, test_labels = train_test_split(merged_img, merged_labels, test_size=0.5, random_state= random_seed, stratify=merged_labels_flat)

train_imgs = train_set.imgs
train_labels = train_set.labels

n = train_imgs.shape[0] + merged_img.shape[0]
print(f"Train: {len(train_imgs):>6} images  ({len(train_imgs)/n:.1%})")
print(f"Val:   {len(val_imgs):>6} images  ({len(val_imgs)/n:.1%})")
print(f"Test:  {len(test_imgs):>6} images  ({len(test_imgs)/n:.1%})")
 
np.savez_compressed(
    "bloodmnist_70_15_15.npz",
    train_images=train_imgs, train_labels=train_labels,
    val_images=val_imgs,     val_labels=val_labels,
    test_images=test_imgs,   test_labels=test_labels,
)
print("Saved custom split -> bloodmnist_70_15_15.npz")