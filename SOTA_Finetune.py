"""
Part 5: SOTA fine-tuning & evaluation on BloodMNIST.

Fine-tunes two pre-trained lightweight networks : MobileNetV2, ShuffleNetV2)
SAME 70/15/15 split used for Model A / Model B
on the test set and reports params, model size (MB), accuracy, macro
precision/recall/F1/AUC, throughput, confusion matrix.

run:  python SOTA_Finetune.py
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torchvision import models

from Data_loader import get_dataloaders
from Train import compute_eval_metrics, plot_loss_curves, train_model

NUM_CLASSES = 8
HEAD_EPOCHS = 5        # phase 1: only the new classifier head is trained
FULL_EPOCHS = 20       # phase 2: whole network fine-tuned 
HEAD_LR = 1e-3
FULL_LR = 1e-4
os.makedirs("Weights", exist_ok=True)


#models
def build_mobilenet_v2():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)   # 1000 -> 8
    return m, m.classifier


def build_shufflenet_v2():
    m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)            # 1000 -> 8
    return m, m.fc


def set_backbone_trainable(model, head, trainable):
    """trainable=False -> only the head learns; True -> everything learns."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head.parameters():
        p.requires_grad = True


def total_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    return total_params(model) * 4 / 1e6          # float32 = 4 bytes/param


#fine-tuning
def finetune(name, builder, train_loader, val_loader, device):
    print(f"\n{'=' * 60}\nFine-tuning {name}\n{'=' * 60}")
    model, head = builder()
    model.to(device)
    # train_model() calls model.count_trainable_parameters();
    model.count_trainable_parameters = lambda m=model: sum(
        p.numel() for p in m.parameters() if p.requires_grad)

    ckpt = f"Weights/{name}.pth"

    # Phase 1: freeze backbone, train new head
    set_backbone_trainable(model, head, False)
    model, history, _, best_f1 = train_model(
        model=model, learning_rate=HEAD_LR, epochs=HEAD_EPOCHS, optimizer_type="adam",
        Save_path=ckpt, train_loader=train_loader, val_loader=val_loader, device=device)

    # Phase 2: unfreeze everything, small LR
    set_backbone_trainable(model, head, True)
    model, history, _, best_f1 = train_model(
        model=model, learning_rate=FULL_LR, epochs=FULL_EPOCHS, optimizer_type="adam",
        Save_path=ckpt, state_dict_path=ckpt, train_loader=train_loader,
        val_loader=val_loader, history=history, best_val_f1=best_f1, device=device)

    plot_loss_curves(history, model, "adam", f"Weights/{name}.png",
                     phase_boundaries=[HEAD_EPOCHS])
    return model, history, ckpt


#evaluation
def evaluate(name, model, ckpt, history, test_loader, device):
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.to(device).eval()

    probs_list, targets_list = [], []
    start = time.time()
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            targets_list.append(labels.numpy())
    elapsed = time.time() - start

    probs = np.concatenate(probs_list)
    targets = np.concatenate(targets_list)
    preds = probs.argmax(1)
    m = compute_eval_metrics(targets, probs)

    result = {
        "model": name,
        "total_params": total_params(model),
        "size_mb_float32": round(model_size_mb(model), 3),
        "size_mb_on_disk": round(os.path.getsize(ckpt) / 1e6, 3),
        "avg_train_time_per_epoch_s": round(float(np.mean(history["epoch_time_sec"])), 2),
        "test_accuracy": round(m["accuracy"], 4),
        "macro_precision": round(m["precision"], 4),
        "macro_recall": round(m["recall"], 4),
        "macro_f1": round(m["f1"], 4),
        "macro_auc": round(m["auc"], 4),
        "throughput_img_per_s": round(len(targets) / elapsed),
    }

    print(f"\n{'=' * 60}\nTEST RESULTS: {name}\n{'=' * 60}")
    for k, v in result.items():
        print(f"{k:28s}: {v}")
    print("\n[Per-class report]")
    print(classification_report(targets, preds, digits=4))
    print("[Confusion matrix]")
    for row in m["confusion_matrix"]:
        print(" ".join(f"{v:4d}" for v in row))

    result["confusion_matrix"] = m["confusion_matrix"].tolist()
    return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=64, num_workers=2)

    results = []
    for name, builder in [("MobileNetV2", build_mobilenet_v2),
                          ("ShuffleNetV2", build_shufflenet_v2)]:
        model, history, ckpt = finetune(name, builder, train_loader, val_loader, device)
        results.append(evaluate(name, model, ckpt, history, test_loader, device))

    with open("Weights/sota_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Weights/sota_results.json")